// Modified from: https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define ENABLE_NOTE_LOG 0

__global__ void flash_attn_1_fwd_f32_kernel(
    const float *Q,
    const float *K,
    const float *V,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float *l,
    float *m,
    float *O)
{ // 1D-Block and 2D-Grid
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index in the grid

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y + by) * N * d; // gridDim.y = nh  qkv dim: (B, nh, N, d)   it equals to (by * gridDim.x + bx) * N * d
    int lm_offset = (bx * gridDim.y + by) * N;      // offset for l and m  lm dim: (B, nh, N)

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d; // size of Qi, Kj, Vj , Bc >= Br, so choose consistent Bc as tile_size
    float *Qi = sram;
    float *Kj = &sram[tile_size];
    float *Vj = &sram[tile_size * 2];
    float *S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++)
    {

// Load Kj, Vj to SRAM
#pragma unroll
        for (int x = 0; x < d; x++) // one block caculates one tile
        {                           // tx * d indexes the thread, x indexes the element in the vector
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads(); // such that the inner loop can use the correct Kj, Vj

#pragma unroll
        for (int i = 0; i < Tr; i++)
        {

// Load Qi to SRAM, l and m to registers
#pragma unroll
            for (int x = 0; x < d; x++)
            { // one thread caculates one row of Qi
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            __syncthreads(); // such that the inner loop can use the correct Qi

            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S) row-wise
            // with causal masking
            float row_m = -INFINITY;
#pragma unroll
            for (int y = 0; y < Bc; y++)
            {
                float sum = 0;
#pragma unroll
                for (int x = 0; x < d; x++)
                {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x]; // a thread caculates one element of S
                }
                sum *= softmax_scale;
                if (i * Br + tx < j * Bc + y)
                    sum = -INFINITY;
                S[(Bc * tx) + y] = sum; // S dim: (Br, Bc)

                if (sum > row_m)
                    row_m = sum; // every thread hold one row_m
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
#pragma unroll
            for (int y = 0; y < Bc; y++)
            {
                if (i * Br + tx < j * Bc + y)
                    S[(Bc * tx) + y] = 0;
                else
                    S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                              (__expf(row_m - row_m_new) * row_l);

// Write O, l, m to HBM
#pragma unroll
            for (int x = 0; x < d; x++)
            {
                float pv = 0; // Pij * Vj
#pragma unroll
                for (int y = 0; y < Bc; y++)
                {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] =
                    (1 / row_l_new) *
                    ((row_l_prev * __expf(row_m_prev - row_m_new) *
                      O[qkv_offset + (tile_size * i) + (tx * d) + x]) +
                     (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads(); // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

__global__ void flash_attn_2_fwd_f32_kernel(
    const float *Q,
    const float *K,
    const float *V,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float *L,
    float *O)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for L

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d; // size of Qi, Kj, Vj
    float *Qi = sram;
    float *Kj = &sram[tile_size];
    float *Vj = &sram[tile_size * 2];
    float *S = &sram[tile_size * 3];

    for (int i = 0; i < Tr; ++i)
    {
        if (i * Br + tx >= N)
            break; // break if we are done with the sequence

        // Load Qi from HBM to SRAM, l and m to registers
        for (int x = 0; x < d; x++)
        {
            Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
        }
        float row_m_prev = -INFINITY;
        float row_l_prev = 0;

        // Causal mask: j <= i
        for (int j = 0; j <= Tc; ++j)
        {
            __syncthreads();
            // Load Kj, Vj from HBM to SRAM
            for (int x = 0; x < d; x++)
            {
                Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
                Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            }
            __syncthreads();
            // S_i^j = softmax_scale * QiKj^T
            // S_i^j[tx][y] = softmax_scale * Sum_{x = 0}^{d-1} Qi[tx][x] * Kj[y][x]
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++)
            {
                if (j * Bc + y >= N)
                    break; // break if we are done with the sequence
                if (i * Br + tx < j * Bc + y)
                    break;
                float sum = 0;
                for (int x = 0; x < d; x++)
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // m_i^j = max(m_i^j-1, row_max(S_i^j))
            float new_row_m = max(row_m_prev, row_m);

            // P_i^j = exp(S_i^j - m_i^j)
            // P_i^j[tx][y] = exp(S_i^j[tx][y] - m_i^j)
            float row_l = 0;
            for (int y = 0; y < Bc; y++)
            {
                if (j * Bc + y >= N)
                    break; // break if we are done with the sequence
                if (i * Br + tx < j * Bc + y)
                    break;
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - new_row_m);
                row_l += S[(Bc * tx) + y];
            }

            // l_i^j = (exp(m_i^j-1 - m_i^j) * l_i^j-1) + row_sum(P_i^j)
            float row_m_exp = __expf(row_m_prev - new_row_m);
            float new_row_l = (row_m_exp * row_l_prev) + row_l;

            // O_i^j = diag(exp(m_i^j-1 - m_i^j))^-1 * O_i^j-1 + P_i^jVj
            for (int x = 0; x < d; x++)
            {
                float pv = 0; // Pij * Vj
                for (int y = 0; y < Bc; y++)
                {
                    if (j * Bc + y >= N)
                        break; // break if we are done with the sequence
                    if (i * Br + tx < j * Bc + y)
                        break;
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] =
                    row_m_exp * O[qkv_offset + (tile_size * i) + (tx * d) + x] + pv;
            }

            // Update m and l
            row_m_prev = new_row_m;
            row_l_prev = new_row_l;
        }

        // O_i = diag(l_i^{Tc})^-1 * O_i^{Tc}
        for (int x = 0; x < d; x++)
            O[qkv_offset + (tile_size * i) + (tx * d) + x] /= row_l_prev;
        // L_i = m_i^{Tc} + log(l_i^{Tc})
        L[lm_offset + (Br * i) + tx] = row_m_prev + __logf(row_l_prev);
    }
}

extern "C"
{

    int flash_attn_1_fwd_f32(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                             void *extra)
    {
        cudaStream_t custream = static_cast<cudaStream_t>(stream);
        if (nparam != 6)
            return 1;
        float *Q = static_cast<float *>(params[0]);
        float *K = static_cast<float *>(params[1]);
        float *V = static_cast<float *>(params[2]);
        // put l,m as input params
        float *l = static_cast<float *>(params[3]);
        float *m = static_cast<float *>(params[4]);
        float *O = static_cast<float *>(params[5]);

        const int B = static_cast<int>(shapes[0][0]);
        const int nh = static_cast<int>(shapes[0][1]);
        const int N = static_cast<int>(shapes[0][2]);
        const int d = static_cast<int>(shapes[0][3]);

        // set block size, TODO: dynamically set block size
        const int Bc = 32;
        const int Br = 32;
        // const int Bc = ceil(max_sram_size / (4 * d * sizeof(float)));
        // const int Br = min(Bc, d);

        const int Tc = ceil((float)N / Bc);
        const int Tr = ceil((float)N / Br);
        const float softmax_scale = 1.0 / sqrt(d);

        // Calculate SRAM size needed per block
        int col_tile_size = Bc * d; // size of Kj, Vj
        int row_tile_size = Br * d; // size of Qi
        const int sram_size =
            (2 * col_tile_size * sizeof(float)) // SRAM size for Kj, Vj
            + (row_tile_size * sizeof(float))   // SRAM size for Qi
            + (Bc * Br * sizeof(float));        // SRAM size for S
        int max_sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

        dim3 grid_dim(B, nh); // batch_size x num_heads
        dim3 block_dim(Bc);   // Bc threads per block

        flash_attn_1_fwd_f32_kernel<<<grid_dim, block_dim, sram_size, custream>>>(
            Q, K, V, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, O);
        return 0;
    }
}

extern "C"
{

    int flash_attn_2_fwd_f32(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                             void *extra)
    {
        cudaStream_t custream = static_cast<cudaStream_t>(stream);
        if (nparam != 5)
            return 1;
        float *Q = static_cast<float *>(params[0]);
        float *K = static_cast<float *>(params[1]);
        float *V = static_cast<float *>(params[2]);
        // put l,m as input params
        float *l = static_cast<float *>(params[3]);
        float *O = static_cast<float *>(params[4]);

        const int B = static_cast<int>(shapes[0][0]);
        const int nh = static_cast<int>(shapes[0][1]);
        const int N = static_cast<int>(shapes[0][2]);
        const int d = static_cast<int>(shapes[0][3]);

        // set block size, TODO: dynamically set block size
        const int Bc = 32;
        const int Br = 32;

        // Calculate SRAM size needed per block
        int col_tile_size = Bc * d; // size of Kj, Vj
        int row_tile_size = Br * d; // size of Qi
        const int sram_size =
            (2 * col_tile_size * sizeof(float)) // SRAM size for Kj, Vj
            + (row_tile_size * sizeof(float))   // SRAM size for Qi
            + (Bc * Br * sizeof(float));        // SRAM size for S
        int max_sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

        const int Tc = ceil((float)N / Bc);
        const int Tr = ceil((float)N / Br);
        const float softmax_scale = 1.0 / sqrt(d);

        printf("Bc: %d, Br: %d, Tc: %d, Tr: %d \n", Bc, Br, Tc, Tr);
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

        dim3 grid_dim(B, nh); // batch_size x num_heads
        dim3 block_dim(Bc);   // Bc threads per block

        flash_attn_2_fwd_f32_kernel<<<grid_dim, block_dim, sram_size, custream>>>(
            Q, K, V, N, d, Tc, Tr, Bc, Br, softmax_scale, l, O);
        return 0;
    }
}
