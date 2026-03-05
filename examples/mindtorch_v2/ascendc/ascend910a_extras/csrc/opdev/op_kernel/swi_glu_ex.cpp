#include "kernel_operator.h"

extern "C" __global__ __aicore__ void swi_glu_ex(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    int num_tokens = tiling_data.num_tokens;
    int dim = tiling_data.dim;
    int core_num = tiling_data.core_num;

    using scalar_t = half;
    using acc_t = float;
    // FIXME: dim should be multiple of BLOCK_SIZE_DIM
    constexpr int BLOCK_SIZE_DIM = 64;
    __gm__ scalar_t *x_ptr = reinterpret_cast<__gm__ scalar_t *>(x);
    __gm__ scalar_t *y_ptr = reinterpret_cast<__gm__ scalar_t *>(y);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x0_que;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x1_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_que;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf;
    AscendC::GlobalTensor<scalar_t> input_tensor;
    AscendC::GlobalTensor<scalar_t> output_tensor;

    // init
    pipe.InitBuffer(x0_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(x1_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(out_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(calc_buf, 5 * BLOCK_SIZE_DIM * sizeof(acc_t));

    for (int64_t i = AscendC::GetBlockIdx(); i < num_tokens; i += core_num) {
        input_tensor.SetGlobalBuffer(x_ptr + dim * 2 * i, dim * 2);
        output_tensor.SetGlobalBuffer(y_ptr + dim * i, dim);

        // FIXME: no bound check
        for (int dim_i = 0; dim_i < dim; dim_i += BLOCK_SIZE_DIM) {
            AscendC::LocalTensor<scalar_t> x0_copy = x0_que.AllocTensor<scalar_t>();
            AscendC::LocalTensor<scalar_t> x1_copy = x1_que.AllocTensor<scalar_t>();
            AscendC::DataCopy(x0_copy, input_tensor[dim_i], BLOCK_SIZE_DIM);
            AscendC::DataCopy(x1_copy, input_tensor[dim + dim_i], BLOCK_SIZE_DIM);
            x0_que.EnQue(x0_copy);
            x1_que.EnQue(x1_copy);


            AscendC::LocalTensor<scalar_t> _y = out_que.AllocTensor<scalar_t>();
            AscendC::LocalTensor<scalar_t> x0 = x0_que.DeQue<scalar_t>();
            AscendC::LocalTensor<scalar_t> x1 = x1_que.DeQue<scalar_t>();
            AscendC::LocalTensor<acc_t> x0_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 0);
            AscendC::LocalTensor<acc_t> x1_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, BLOCK_SIZE_DIM * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> sigmoid_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 2 * BLOCK_SIZE_DIM * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> prod_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 3 * BLOCK_SIZE_DIM * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> mul_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 4 * BLOCK_SIZE_DIM * sizeof(acc_t));
            Cast(x0_f32, x0, AscendC::RoundMode::CAST_NONE, BLOCK_SIZE_DIM);
            Cast(x1_f32, x1, AscendC::RoundMode::CAST_NONE, BLOCK_SIZE_DIM);
            Sigmoid(sigmoid_f32, x0_f32, BLOCK_SIZE_DIM);
            Mul(prod_f32, x0_f32, sigmoid_f32, BLOCK_SIZE_DIM);
            Mul(mul_f32, prod_f32, x1_f32, BLOCK_SIZE_DIM);
            Cast(_y, mul_f32, AscendC::RoundMode::CAST_ODD, BLOCK_SIZE_DIM);
            out_que.EnQue(_y);
            x0_que.FreeTensor(x0);
            x1_que.FreeTensor(x1);

            AscendC::LocalTensor<scalar_t> y_copy = out_que.DeQue<scalar_t>();
            AscendC::DataCopy(output_tensor[dim_i], y_copy, BLOCK_SIZE_DIM);
            out_que.FreeTensor(y_copy);
        }
    }

}

