// Modified from: https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu  
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define ENABLE_NOTE_LOG 0

__global__ void flash_attn_2_fwd_f32_kernel(
  const float* Q, 
  const float* K, 
  const float* V, 
  const int N, 
  const int d,
  const int Tc,
  const int Tr, 
  const int Bc, 
  const int Br, 
  const float softmax_scale,
  float* l, 
  float *m, 
  float* O) {
  int tx = threadIdx.x;
  int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

  // Offset into Q,K,V,O,l,m - different for each batch and head
  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
  int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

  // Define SRAM for Q,K,V,S
  extern __shared__ float sram[];
  int tile_size = Bc * d;  // size of Qi, Kj, Vj
  float* Qi = sram;
  float* Kj = &sram[tile_size];
  float* Vj = &sram[tile_size * 2];
  float* S = &sram[tile_size * 3];
  
  // TODO: swap the load order of Kj, Vj and Qi. first load Qi, then Kj, Vj
  for (int j = 0; j < Tc; j++) {

      // Load Kj, Vj to SRAM
      for (int x = 0; x < d; x++) {
          Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
          Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
      }
      __syncthreads();  // such that the inner loop can use the correct Kj, Vj

      for (int i = 0; i < Tr; i++)  {

          // Load Qi to SRAM, l and m to registers
          for (int x = 0; x < d; x++) {
              Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
          }
          float row_m_prev = m[lm_offset + (Br * i) + tx];
          float row_l_prev = l[lm_offset + (Br * i) + tx];

          // S = QK^T, row_m = rowmax(S)
          float row_m = -INFINITY;
          for (int y = 0; y < Bc; y++) {
              float sum = 0;
              for (int x = 0; x < d; x++) {
                  sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
              }
              sum *= softmax_scale;
              S[(Bc * tx) + y] = sum;

              if (sum > row_m)
                  row_m = sum;
          }

          // P = exp(S - row_m), row_l = rowsum(P)
          float row_l = 0;
          for (int y = 0; y < Bc; y++) {
              S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
              row_l += S[(Bc * tx) + y];
          }

          // Compute new m and l
          float row_m_new = max(row_m_prev, row_m);
          float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

          // Write O, l, m to HBM
          for (int x = 0; x < d; x++) {
              float pv = 0;  // Pij * Vj
              for (int y = 0; y < Bc; y++) {
                  pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
              }
              O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                  * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                  + (__expf(row_m - row_m_new) * pv));
          }
          m[lm_offset + (Br * i) + tx] = row_m_new;
          l[lm_offset + (Br * i) + tx] = row_l_new;
      }
      __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
  }
}


extern "C" {

    int flash_forward(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                    void *extra) {
        cudaStream_t custream = static_cast<cudaStream_t>(stream);
        if (nparam != 6) return 1;
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

        int max_sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

        // set block size, TODO: dynamically set block size
        const int Bc = 32;
        const int Br = 32;

        const int Tc = ceil((float) N / Bc);
        const int Tr = ceil((float) N / Br);
        const float softmax_scale = 1.0 / sqrt(d);

        // Calculate SRAM size needed per block
        const int sram_size = (2 * Bc * d * sizeof(float)) + (4 * Br * d * sizeof(float));
        printf("Bc: %d, Br: %d, Tc: %d, Tr: %d \n", Bc, Br, Tc, Tr);
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

        dim3 grid_dim(B, nh);  // batch_size x num_heads
        dim3 block_dim(Bc);  // Bc threads per block

        flash_attn_2_fwd_f32_kernel<<<grid_dim, block_dim, sram_size, custream>>>(
            Q, K, V, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, O
        );
        return 0;
    }
}