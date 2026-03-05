#include "kernel_operator.h"

extern "C" __global__ __aicore__ void add_rms_norm_ex(
    GM_ADDR x, GM_ADDR residual, GM_ADDR weight, GM_ADDR epsilon,
    GM_ADDR y, GM_ADDR residual_output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    int num_tokens = tiling_data.num_tokens;
    int dim = tiling_data.dim;
    int core_num = tiling_data.core_num;

    using scalar_t = half;
    using acc_t = float;
    constexpr int BLOCK_SIZE_DIM = 64;

    __gm__ scalar_t *x_ptr = reinterpret_cast<__gm__ scalar_t *>(x);
    __gm__ scalar_t *r_ptr = reinterpret_cast<__gm__ scalar_t *>(residual);
    __gm__ scalar_t *w_ptr = reinterpret_cast<__gm__ scalar_t *>(weight);
    __gm__ scalar_t *y_ptr = reinterpret_cast<__gm__ scalar_t *>(y);
    __gm__ scalar_t *r_out_ptr = reinterpret_cast<__gm__ scalar_t *>(residual_output);
    __gm__ float* epsilon_ptr = reinterpret_cast<__gm__ float*>(epsilon);
    acc_t epsilon_val = static_cast<acc_t>(*epsilon_ptr);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x_que;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> r_que;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> w_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> r_out_que;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf;
    AscendC::GlobalTensor<scalar_t> input_tensor;
    AscendC::GlobalTensor<scalar_t> residual_tensor;
    AscendC::GlobalTensor<scalar_t> weight_tensor;
    AscendC::GlobalTensor<scalar_t> output_tensor;
    AscendC::GlobalTensor<scalar_t> residual_output_tensor;

    pipe.InitBuffer(x_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(r_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(w_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(out_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(r_out_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(calc_buf, 5 * BLOCK_SIZE_DIM * sizeof(acc_t));

    for (int64_t i = AscendC::GetBlockIdx(); i < num_tokens; i += core_num) {
        input_tensor.SetGlobalBuffer(x_ptr + dim * i, dim);
        residual_tensor.SetGlobalBuffer(r_ptr + dim * i, dim);
        weight_tensor.SetGlobalBuffer(w_ptr, dim);
        output_tensor.SetGlobalBuffer(y_ptr + dim * i, dim);
        residual_output_tensor.SetGlobalBuffer(r_out_ptr + dim * i, dim);

        // 1. 第一遍遍历，累加 sum_sq
        acc_t sum_sq = 0.0f;
        for (int dim_i = 0; dim_i < dim; dim_i += BLOCK_SIZE_DIM) {
            int curr_block_size = (dim_i + BLOCK_SIZE_DIM <= dim) ? BLOCK_SIZE_DIM : (dim - dim_i);
            AscendC::LocalTensor<scalar_t> x_copy = x_que.AllocTensor<scalar_t>();
            AscendC::DataCopy(x_copy, input_tensor[dim_i], curr_block_size);
            x_que.EnQue(x_copy);
            AscendC::LocalTensor<scalar_t> r_copy = r_que.AllocTensor<scalar_t>();
            AscendC::DataCopy(r_copy, residual_tensor[dim_i], curr_block_size);
            r_que.EnQue(r_copy);
            AscendC::LocalTensor<scalar_t> x = x_que.DeQue<scalar_t>();
            AscendC::LocalTensor<scalar_t> r = r_que.DeQue<scalar_t>();
            AscendC::LocalTensor<acc_t> x_f32 = calc_buf.GetWithOffset<acc_t>(curr_block_size, 0);
            AscendC::LocalTensor<acc_t> r_f32 = calc_buf.GetWithOffset<acc_t>(curr_block_size, curr_block_size * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> tmp = calc_buf.GetWithOffset<acc_t>(curr_block_size, 2 * curr_block_size * sizeof(acc_t));
            Cast(x_f32, x, AscendC::RoundMode::CAST_NONE, curr_block_size);
            Cast(r_f32, r, AscendC::RoundMode::CAST_NONE, curr_block_size);
            Add(tmp, x_f32, r_f32, curr_block_size);
            for (int j = 0; j < curr_block_size; ++j) {
                acc_t v = tmp.GetValue(j);
                sum_sq += v * v;
            }
            x_que.FreeTensor(x);
            r_que.FreeTensor(r);
        }
        acc_t rms = sqrt(sum_sq / dim + static_cast<acc_t>(*epsilon_ptr));

        // 2. 第二遍遍历，归一化和写回
        for (int dim_i = 0; dim_i < dim; dim_i += BLOCK_SIZE_DIM) {
            int curr_block_size = (dim_i + BLOCK_SIZE_DIM <= dim) ? BLOCK_SIZE_DIM : (dim - dim_i);
            AscendC::LocalTensor<scalar_t> x_copy = x_que.AllocTensor<scalar_t>();
            AscendC::DataCopy(x_copy, input_tensor[dim_i], curr_block_size);
            x_que.EnQue(x_copy);
            AscendC::LocalTensor<scalar_t> r_copy = r_que.AllocTensor<scalar_t>();
            AscendC::DataCopy(r_copy, residual_tensor[dim_i], curr_block_size);
            r_que.EnQue(r_copy);
            AscendC::LocalTensor<scalar_t> w_copy = w_que.AllocTensor<scalar_t>();
            AscendC::DataCopy(w_copy, weight_tensor[dim_i], curr_block_size);
            w_que.EnQue(w_copy);
            AscendC::LocalTensor<scalar_t> x = x_que.DeQue<scalar_t>();
            AscendC::LocalTensor<scalar_t> r = r_que.DeQue<scalar_t>();
            AscendC::LocalTensor<scalar_t> w = w_que.DeQue<scalar_t>();
            AscendC::LocalTensor<scalar_t> y_copy = out_que.AllocTensor<scalar_t>();
            AscendC::LocalTensor<scalar_t> r_out_copy = r_out_que.AllocTensor<scalar_t>();
            AscendC::LocalTensor<acc_t> x_f32 = calc_buf.GetWithOffset<acc_t>(curr_block_size, 0);
            AscendC::LocalTensor<acc_t> r_f32 = calc_buf.GetWithOffset<acc_t>(curr_block_size, curr_block_size * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> w_f32 = calc_buf.GetWithOffset<acc_t>(curr_block_size, 2 * curr_block_size * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> tmp = calc_buf.GetWithOffset<acc_t>(curr_block_size, 3 * curr_block_size * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> norm_tmp = calc_buf.GetWithOffset<acc_t>(curr_block_size, 4 * curr_block_size * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> y_f32 = calc_buf.GetWithOffset<acc_t>(curr_block_size, 0); // reuse buffer
            Cast(x_f32, x, AscendC::RoundMode::CAST_NONE, curr_block_size);
            Cast(r_f32, r, AscendC::RoundMode::CAST_NONE, curr_block_size);
            Cast(w_f32, w, AscendC::RoundMode::CAST_NONE, curr_block_size);
            Add(tmp, x_f32, r_f32, curr_block_size);
            for (int j = 0; j < curr_block_size; ++j) {
                norm_tmp.SetValue(j, tmp.GetValue(j) / rms);
            }
            Mul(y_f32, norm_tmp, w_f32, curr_block_size);
            Cast(y_copy, y_f32, AscendC::RoundMode::CAST_NONE, curr_block_size);
            out_que.EnQue(y_copy);
            Cast(r_out_copy, tmp, AscendC::RoundMode::CAST_NONE, curr_block_size);
            r_out_que.EnQue(r_out_copy);
            x_que.FreeTensor(x);
            r_que.FreeTensor(r);
            w_que.FreeTensor(w);
            AscendC::LocalTensor<scalar_t> y = out_que.DeQue<scalar_t>();
            AscendC::DataCopy(output_tensor[dim_i], y, curr_block_size);
            out_que.FreeTensor(y);
            AscendC::LocalTensor<scalar_t> r_out = r_out_que.DeQue<scalar_t>();
            AscendC::DataCopy(residual_output_tensor[dim_i], r_out, curr_block_size);
            r_out_que.FreeTensor(r_out);
        }
    }
}

