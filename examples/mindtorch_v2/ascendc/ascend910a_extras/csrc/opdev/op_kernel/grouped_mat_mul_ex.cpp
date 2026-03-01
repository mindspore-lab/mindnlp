#include "kernel_operator.h"

#include "matmul_core.h"

extern "C" __global__ __aicore__ void grouped_mat_mul_ex(GM_ADDR x, GM_ADDR w, GM_ADDR group_list, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    using scalar_t = half;
    using acc_t = float;
    using index_t = int64_t;

    int num_tokens = tiling_data.num_tokens;
    int dim = tiling_data.dim;
    int num_exports = tiling_data.num_exports;
    int inner_dim = tiling_data.inner_dim;
    int core_num = tiling_data.core_num;

    __gm__ scalar_t *x_ptr = reinterpret_cast<__gm__ scalar_t *>(x);
    __gm__ scalar_t *w_ptr = reinterpret_cast<__gm__ scalar_t *>(w);
    __gm__ scalar_t *y_ptr = reinterpret_cast<__gm__ scalar_t *>(y);
    __gm__ index_t *group_list_ptr = reinterpret_cast<__gm__ index_t *>(group_list);


    MatMulNT<scalar_t, acc_t, index_t> matmul;
    matmul.InitPipe();

    AscendC::GlobalTensor<index_t> offset;
    offset.SetGlobalBuffer(group_list_ptr, num_exports);
    for (int ei = AscendC::GetBlockIdx(); ei < num_exports; ei += core_num) {
        index_t start = 0;
        index_t end = 0;
        if (ei == 0) {
            start = 0;
            end = offset.GetValue(ei);
        } else {
            start = offset.GetValue(ei - 1);
            end = offset.GetValue(ei);
        }
        index_t curr_num_tokens = end - start;
        if (curr_num_tokens <= 0) continue;
        matmul.InitSize(curr_num_tokens, inner_dim, dim);
        matmul.InitBuffer(x_ptr + start * dim, w_ptr + ei * dim * inner_dim, y_ptr + start * inner_dim);
        matmul.Process();
    }
}

