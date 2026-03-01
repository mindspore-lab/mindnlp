#include "kernel_operator.h"

extern "C" __global__ __aicore__ void rope_ex(GM_ADDR q, GM_ADDR k, GM_ADDR position_ids, GM_ADDR cos_cache, GM_ADDR sin_cache, GM_ADDR out_q, GM_ADDR out_k, GM_ADDR workspace, GM_ADDR tiling) {
    using scalar_t = half;
    using index_t = int32_t;

    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    int num_heads = tiling_data.num_heads;
    int num_kv_heads = tiling_data.num_kv_heads;
    int head_dim = tiling_data.head_dim;
    int embed_dim = head_dim / 2;

    int core_num = AscendC::GetBlockNum();
    int batch_id = AscendC::GetBlockIdx();

    __gm__ scalar_t *q_ptr = reinterpret_cast<__gm__ scalar_t *>(q);
    __gm__ scalar_t *k_ptr = reinterpret_cast<__gm__ scalar_t *>(k);
    __gm__ index_t *position_ids_ptr = reinterpret_cast<__gm__ index_t *>(position_ids);
    __gm__ scalar_t *cos_cache_ptr = reinterpret_cast<__gm__ scalar_t *>(cos_cache);
    __gm__ scalar_t *sin_cache_ptr = reinterpret_cast<__gm__ scalar_t *>(sin_cache);
    __gm__ scalar_t *out_q_ptr = reinterpret_cast<__gm__ scalar_t *>(out_q);
    __gm__ scalar_t *out_k_ptr = reinterpret_cast<__gm__ scalar_t *>(out_k);


    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> q_que;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> k_que;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> cos_cache_que;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> sin_cache_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_q_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_k_que;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf;

    AscendC::GlobalTensor<scalar_t> q_gm;
    AscendC::GlobalTensor<scalar_t> k_gm;
    AscendC::GlobalTensor<index_t> position_ids_gm;
    AscendC::GlobalTensor<scalar_t> cos_cache_gm;
    AscendC::GlobalTensor<scalar_t> sin_cache_gm;
    AscendC::GlobalTensor<scalar_t> out_q_gm;
    AscendC::GlobalTensor<scalar_t> out_k_gm;

    pipe.InitBuffer(q_que, 1, sizeof(scalar_t) * num_heads * head_dim);
    pipe.InitBuffer(k_que, 1, sizeof(scalar_t) * num_kv_heads * head_dim);
    pipe.InitBuffer(cos_cache_que, 1, sizeof(scalar_t) * embed_dim);
    pipe.InitBuffer(sin_cache_que, 1, sizeof(scalar_t) * embed_dim);
    pipe.InitBuffer(out_q_que, 1, sizeof(scalar_t) * num_heads * head_dim);
    pipe.InitBuffer(out_k_que, 1, sizeof(scalar_t) * num_kv_heads * head_dim);
    pipe.InitBuffer(calc_buf, 4 * sizeof(scalar_t) * embed_dim);

    position_ids_gm.SetGlobalBuffer(position_ids_ptr + batch_id, 1);
    int pos = position_ids_gm.GetValue(0);

    q_gm.SetGlobalBuffer(q_ptr + batch_id * num_heads * head_dim, num_heads * head_dim);
    k_gm.SetGlobalBuffer(k_ptr + batch_id * num_kv_heads * head_dim, num_kv_heads * head_dim);

    // cos_cache: [_, embed_dim]
    cos_cache_gm.SetGlobalBuffer(cos_cache_ptr + pos * embed_dim, embed_dim);
    sin_cache_gm.SetGlobalBuffer(sin_cache_ptr + pos * embed_dim, embed_dim);

    out_q_gm.SetGlobalBuffer(out_q_ptr + batch_id * num_heads * head_dim, num_heads * head_dim);
    out_k_gm.SetGlobalBuffer(out_k_ptr + batch_id * num_kv_heads * head_dim, num_kv_heads * head_dim);

    // load q/k and cos/sin
    AscendC::LocalTensor<scalar_t> q_copy = q_que.AllocTensor<scalar_t>();
    AscendC::LocalTensor<scalar_t> k_copy = k_que.AllocTensor<scalar_t>();
    AscendC::DataCopy(q_copy, q_gm, num_heads * head_dim);
    AscendC::DataCopy(k_copy, k_gm, num_kv_heads * head_dim);
    q_que.EnQue(q_copy);
    k_que.EnQue(k_copy);
    AscendC::LocalTensor<scalar_t> cos_copy = cos_cache_que.AllocTensor<scalar_t>();
    AscendC::LocalTensor<scalar_t> sin_copy = sin_cache_que.AllocTensor<scalar_t>();
    AscendC::DataCopy(cos_copy, cos_cache_gm, embed_dim);
    AscendC::DataCopy(sin_copy, sin_cache_gm, embed_dim);
    cos_cache_que.EnQue(cos_copy);
    sin_cache_que.EnQue(sin_copy);

    // compute
    AscendC::LocalTensor<scalar_t> x0 = calc_buf.GetWithOffset<scalar_t>(embed_dim, 0);
    AscendC::LocalTensor<scalar_t> x1 = calc_buf.GetWithOffset<scalar_t>(embed_dim, embed_dim * sizeof(scalar_t));
    AscendC::LocalTensor<scalar_t> x2 = calc_buf.GetWithOffset<scalar_t>(embed_dim, 2 * embed_dim * sizeof(scalar_t));
    AscendC::LocalTensor<scalar_t> x3 = calc_buf.GetWithOffset<scalar_t>(embed_dim, 3 * embed_dim * sizeof(scalar_t));
    AscendC::LocalTensor<scalar_t> cos_local = cos_cache_que.DeQue<scalar_t>();
    AscendC::LocalTensor<scalar_t> sin_local = sin_cache_que.DeQue<scalar_t>();

    AscendC::LocalTensor<scalar_t> q_local = q_que.DeQue<scalar_t>();
    AscendC::LocalTensor<scalar_t> out_q_local = out_q_que.AllocTensor<scalar_t>();
    for (int i = 0; i < num_heads; ++i) {
        Mul(x0, q_local[i * head_dim], cos_local, embed_dim);
        Mul(x1, q_local[i * head_dim + embed_dim], sin_local, embed_dim);
        Sub(out_q_local[i * head_dim], x0, x1, embed_dim);
        Mul(x2, q_local[i * head_dim + embed_dim], cos_local, embed_dim);
        Mul(x3, q_local[i * head_dim], sin_local, embed_dim);
        Add(out_q_local[i * head_dim + embed_dim], x2, x3, embed_dim);
    }
    out_q_que.EnQue(out_q_local);
    q_que.FreeTensor(q_local);

    AscendC::LocalTensor<scalar_t> k_local = k_que.DeQue<scalar_t>();
    AscendC::LocalTensor<scalar_t> out_k_local = out_k_que.AllocTensor<scalar_t>();
    for (int i = 0; i < num_kv_heads; ++i) {
        Mul(x0, k_local[i * head_dim], cos_local, embed_dim);
        Mul(x1, k_local[i * head_dim + embed_dim], sin_local, embed_dim);
        Sub(out_k_local[i * head_dim], x0, x1, embed_dim);
        Mul(x2, k_local[i * head_dim + embed_dim], cos_local, embed_dim);
        Mul(x3, k_local[i * head_dim], sin_local, embed_dim);
        Add(out_k_local[i * head_dim + embed_dim], x2, x3, embed_dim);
    }
    out_k_que.EnQue(out_k_local);
    k_que.FreeTensor(k_local);

    cos_cache_que.FreeTensor(cos_local);
    sin_cache_que.FreeTensor(sin_local);

    // store
    AscendC::LocalTensor<scalar_t> out_q_copy = out_q_que.DeQue<scalar_t>();
    AscendC::LocalTensor<scalar_t> out_k_copy = out_k_que.DeQue<scalar_t>();
    AscendC::DataCopy(out_q_gm, out_q_copy, num_heads * head_dim);
    AscendC::DataCopy(out_k_gm, out_k_copy, num_kv_heads * head_dim);
    out_q_que.FreeTensor(out_q_copy);
    out_k_que.FreeTensor(out_k_copy);
}

