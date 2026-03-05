#include "kernel_operator.h"

template<typename scalar_t, typename acc_t>
class PagedAttention {
public:
    static constexpr int BLOCK_M = 16;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::A1, 1> q_a1_que, p_a1_que;
    AscendC::TQue<AscendC::QuePosition::A2, 1> q_a2_que, p_a2_que;
    AscendC::TQue<AscendC::QuePosition::B1, 1> k_b1_que, v_b1_que;
    AscendC::TQue<AscendC::QuePosition::B2, 1> k_b2_que, v_b2_que;
    AscendC::TQue<AscendC::QuePosition::CO1, 1> s_co1_que, o_co1_que;
    AscendC::TQue<AscendC::QuePosition::CO2, 1> s_co2_que, o_co2_que;

    AscendC::TQue<AscendC::QuePosition::VECCALC, 1> s_que;
    AscendC::TQue<AscendC::QuePosition::VECCALC, 1> p_f32_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> p_que, o_que;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> row_max_buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> row_sum_buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> o_scale_buf;

    AscendC::GlobalTensor<scalar_t> q_gm;
    AscendC::GlobalTensor<scalar_t> key_cache_gm;
    AscendC::GlobalTensor<scalar_t> value_cache_gm;
    AscendC::GlobalTensor<int32_t> block_tables_gm;
    AscendC::GlobalTensor<int32_t> context_lens_gm;
    AscendC::GlobalTensor<scalar_t> o_gm;

    // current batch id and kv head id
    int batch_id;
    int kv_head_id;

    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t page_size;
    uint32_t group_size;
    uint32_t max_page_num_per_seq;
    float scale;
    float scale_log2;

    int64_t stride_qo_bs;
    int64_t stride_qo_h;
    int64_t stride_qo_d;
    int64_t stride_kv_p;
    int64_t stride_kv_h;
    int64_t stride_tables_bs;

    __aicore__ inline PagedAttention(
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t page_size,
        uint32_t max_page_num_per_seq,
        float scale,
        float scale_log2
    ) {
        this->num_heads = num_heads;
        this->num_kv_heads = num_kv_heads;
        this->head_dim = head_dim;
        this->page_size = page_size;
        this->group_size = num_heads / num_kv_heads;
        this->max_page_num_per_seq = max_page_num_per_seq;
        this->scale = scale;
        this->scale_log2 = scale_log2;
    }
    __aicore__ inline void Init(
        __gm__ scalar_t *q,
        __gm__ scalar_t *key_cache,
        __gm__ scalar_t *value_cache,
        __gm__ int32_t *block_tables,
        __gm__ int32_t *context_lens,
        __gm__ scalar_t *o,
        int64_t stride_qo_bs,
        int64_t stride_qo_h,
        int64_t stride_qo_d,
        int64_t stride_kv_p,
        int64_t stride_kv_h,
        int64_t stride_tables_bs
    ) {
        this->stride_qo_bs = stride_qo_bs;
        this->stride_qo_h = stride_qo_h;
        this->stride_qo_d = stride_qo_d;
        this->stride_kv_p = stride_kv_p;
        this->stride_kv_h = stride_kv_h;
        this->stride_tables_bs = stride_tables_bs;

        int core_id = AscendC::GetBlockIdx();
        batch_id = core_id / num_kv_heads;
        kv_head_id = core_id % num_kv_heads;

        q_gm.SetGlobalBuffer(q + batch_id * stride_qo_bs + kv_head_id * group_size * stride_qo_h, group_size * head_dim);
        o_gm.SetGlobalBuffer(o + batch_id * stride_qo_bs + kv_head_id * group_size * stride_qo_h, group_size * head_dim);
        key_cache_gm.SetGlobalBuffer(key_cache + kv_head_id * stride_kv_h);
        value_cache_gm.SetGlobalBuffer(value_cache + kv_head_id * stride_kv_h);
        block_tables_gm.SetGlobalBuffer(block_tables + batch_id * stride_tables_bs, max_page_num_per_seq);
        context_lens_gm.SetGlobalBuffer(context_lens + batch_id, 1);

        pipe.InitBuffer(q_a1_que, 1, group_size * head_dim * sizeof(scalar_t));
        pipe.InitBuffer(q_a2_que, 1, BLOCK_M * head_dim * sizeof(scalar_t));
        pipe.InitBuffer(k_b1_que, 1, page_size * head_dim * sizeof(scalar_t));
        pipe.InitBuffer(k_b2_que, 1, page_size * head_dim * sizeof(scalar_t));
        pipe.InitBuffer(s_co1_que, 1, BLOCK_M * page_size * sizeof(acc_t));
        pipe.InitBuffer(s_co2_que, 1, group_size * page_size * sizeof(acc_t));

        pipe.InitBuffer(p_a1_que, 1, group_size * page_size * sizeof(scalar_t));
        pipe.InitBuffer(p_a2_que, 1, BLOCK_M * page_size * sizeof(scalar_t));
        pipe.InitBuffer(v_b1_que, 1, page_size * head_dim * sizeof(scalar_t));
        pipe.InitBuffer(v_b2_que, 1, page_size * head_dim * sizeof(scalar_t));
        pipe.InitBuffer(o_co1_que, 1, BLOCK_M * head_dim * sizeof(acc_t));
        pipe.InitBuffer(o_co2_que, 1, group_size * head_dim * sizeof(acc_t));

        pipe.InitBuffer(s_que, 1, group_size * page_size * sizeof(acc_t));
        pipe.InitBuffer(p_f32_que, 1, group_size * page_size * sizeof(acc_t));
        pipe.InitBuffer(p_que, 1, group_size * page_size * sizeof(scalar_t));
        pipe.InitBuffer(o_que, 1, group_size * head_dim * sizeof(scalar_t));

        pipe.InitBuffer(row_max_buf, group_size * sizeof(acc_t));
        pipe.InitBuffer(row_sum_buf, group_size * sizeof(acc_t));
        pipe.InitBuffer(o_scale_buf, group_size * sizeof(acc_t));
    }

    __aicore__ inline void Process() {
        LoadQ();
        AscendC::LocalTensor<scalar_t> q_a2 = q_a2_que.DeQue<scalar_t>();

        int32_t seq_len = context_lens_gm.GetValue(0);
        int cur_page_num = (seq_len + page_size - 1) / page_size;
        AscendC::LocalTensor<acc_t> row_max = row_max_buf.Get<acc_t>(group_size);
        AscendC::LocalTensor<acc_t> row_sum = row_sum_buf.Get<acc_t>(group_size);
        AscendC::LocalTensor<acc_t> o_scale = o_scale_buf.Get<acc_t>(group_size);
        InitStates(row_max, row_sum, o_scale);

        AscendC::LocalTensor<acc_t> o_co1 = o_co1_que.AllocTensor<acc_t>();
        for (int i = 0; i < cur_page_num; i++) {
            int32_t page_id = block_tables_gm.GetValue(i);
            LoadK(page_id);
            LoadV(page_id);
        //     // gemm qk
        //     MmaQK(q_a2);
        //     CopySFromCO1ToCO2<true>();
        //     // softmax
        //     Softmax<true>(row_max, row_sum, o_scale);
        //     LoadP();
        //     // gemm pv
        //     MmaPV(o_co1, true);
        }
        q_a2_que.FreeTensor(q_a2);
        o_co1_que.EnQue(o_co1);
        // StoreO();
    }
    __aicore__ inline void InitStates(
        AscendC::LocalTensor<acc_t>& row_max,
        AscendC::LocalTensor<acc_t>& row_sum,
        AscendC::LocalTensor<acc_t>& o_scale
    ) {
        Duplicate(row_max, -5e4f, group_size);
        Duplicate(row_sum, 0.0f, group_size);
        Duplicate(o_scale, 1.0f, group_size);
    }
    __aicore__ inline void LoadQ() {
        // gm -> a1
        {
            AscendC::LocalTensor<scalar_t> q_a1 = q_a1_que.AllocTensor<scalar_t>();
            // q: nd -> nz
            for (int i = 0; i < group_size / 16; ++i) {
                int src_offset = i * 16;
                int dst_offset = i * 16 * group_size;
                AscendC::DataCopy(q_a1[dst_offset], q_gm[src_offset], { (uint16_t)group_size, 1, uint16_t(head_dim / 16 - 1), 0 });
            }
            q_a1_que.EnQue(q_a1);
        }
        // a1 -> a2
        {
            AscendC::LocalTensor<scalar_t> q_a2 = q_a2_que.AllocTensor<scalar_t>();
            AscendC::LocalTensor<scalar_t> q_a1 = q_a1_que.DeQue<scalar_t>();
            // q: nz -> zz
            for (int i = 0; i < group_size / 16; ++i) {
                int src_offset = i * 16 * 16;
                int dst_offset = i * 16 * head_dim;
                AscendC::LoadData2dParams params;
                params.repeatTimes = head_dim / 16;
                params.srcStride = group_size / 16;
                params.ifTranspose = false;
                AscendC::LoadData(q_a2[dst_offset], q_a1[src_offset], params);
            }
            q_a2_que.EnQue(q_a2);
            q_a1_que.FreeTensor(q_a1);
        }
    }
    __aicore__ inline void LoadK(int32_t page_id) {
        int page_offset = page_id * stride_kv_p;
        // gm -> b1
        {
            // k: [head_dim / 16, page_size, 16]
            // already zn
            AscendC::LocalTensor<scalar_t> k_b1 = k_b1_que.AllocTensor<scalar_t>();
            AscendC::DataCopy(k_b1, key_cache_gm[page_offset], {1, (uint16_t)(head_dim * page_size / 16), 0, 0});
            k_b1_que.EnQue(k_b1);
        }
        // b1 -> b2
        {
            AscendC::LocalTensor<scalar_t> k_b2 = k_b2_que.AllocTensor<scalar_t>();
            AscendC::LocalTensor<scalar_t> k_b1 = k_b1_que.DeQue<scalar_t>();
            AscendC::LoadData2dParams params;
            params.repeatTimes = (page_size * head_dim) / (16 * 16);
            params.srcStride = 1;
            params.ifTranspose = false;
            AscendC::LoadData(k_b2, k_b1, params);
            k_b2_que.EnQue(k_b2);
            k_b1_que.FreeTensor(k_b1);
        }
    }

    __aicore__ inline void LoadV(int32_t page_id) {
        int page_offset = page_id * stride_kv_p;
        // gm -> b1
        {
            // k: [head_dim / 16, page_size, 16]
            // nz -> zz
            AscendC::LocalTensor<scalar_t> v_b1 = v_b1_que.AllocTensor<scalar_t>();
            for (int i = 0; i < page_size / 16; ++i) {
                int src_offset = i * 16 * 16;
                int dst_offset = i * 16 * head_dim;
                AscendC::DataCopyParams params;
                params.blockCount = head_dim / 16;
                params.blockLen = 16; // 16 * 16 * 2B = 512B = 32B * 16
                params.srcStride = (page_size / 16 - 1) * 16;
                params.dstStride = 0;
                AscendC::DataCopy(v_b1[dst_offset], value_cache_gm[page_offset + src_offset], params);
            }
            v_b1_que.EnQue(v_b1);
        }
        // b1 -> b2
        {
            // zz -> zn
            AscendC::LocalTensor<scalar_t> v_b2 = v_b2_que.AllocTensor<scalar_t>();
            AscendC::LocalTensor<scalar_t> v_b1 = v_b1_que.DeQue<scalar_t>();
            AscendC::LoadData2dParams params;
            params.repeatTimes = (page_size * head_dim) / (16 * 16);
            params.srcStride = 1;
            params.ifTranspose = true;
            AscendC::LoadData(v_b2, v_b1, params);
            v_b2_que.EnQue(v_b2);
            v_b1_que.FreeTensor(v_b1);
        }
    }

    __aicore__ inline void MmaQK(AscendC::LocalTensor<scalar_t>& q_a2) {
        // q_a2: [BLOCK_M, head_dim]
        // k_b2: [page_size, head_dim]
        // s_co1: [BLOCK_M, page_size]
        AscendC::LocalTensor<scalar_t> k_b2 = k_b2_que.DeQue<scalar_t>();
        AscendC::LocalTensor<acc_t> s_co1 = s_co1_que.AllocTensor<acc_t>();
        AscendC::MmadParams params;
        params.m = BLOCK_M;
        params.n = page_size;
        params.k = head_dim;
        params.cmatrixInitVal = true;
        AscendC::Mmad(s_co1, q_a2, k_b2, params);
        s_co1_que.EnQue(s_co1);
        k_b2_que.FreeTensor(k_b2);
    }

    template<bool NzToNd>
    __aicore__ inline void CopySFromCO1ToCO2() {
        AscendC::LocalTensor<acc_t> s_co1 = s_co1_que.DeQue<acc_t>();
        AscendC::LocalTensor<acc_t> s_co2 = s_co2_que.AllocTensor<acc_t>();
        if (NzToNd) {
            // nz -> nd
            for (int i = 0; i < head_dim / 16; ++i) {
                int src_offset = i * group_size * 16;
                int dst_offset = i * 16;
                // 32B
                AscendC::DataCopyParams params;
                params.blockCount = group_size;
                params.blockLen = 2; // 16 * f32 = 64B = 2 * 32B
                params.srcStride = 0;
                params.dstStride = (head_dim / 16 - 1) * 2;
                AscendC::DataCopy(s_co2[dst_offset], s_co1[src_offset], params);
                // AscendC::DataCopyEnhancedParams enhanced_params;
                // enhanced_params.blockMode = AscendC::BlockMode::BLOCK_MODE_VECTOR;
                // AscendC::DataCopy(s_co2[dst_offset], s_co1[src_offset], params, enhanced_params);
            }
        } else {
            // nz -> nz
            AscendC::DataCopyParams params;
            params.blockCount = 1;
            params.blockLen = (group_size * page_size) / (16 * 16);
            AscendC::DataCopyEnhancedParams enhanced_params;
            enhanced_params.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
            AscendC::DataCopy(s_co2, s_co1, params, enhanced_params);
        }
        s_co2_que.EnQue(s_co2);
        s_co1_que.FreeTensor(s_co1);
    }

    template<bool isNd>
    __aicore__ inline void Softmax(AscendC::LocalTensor<acc_t>& row_max, AscendC::LocalTensor<acc_t>& row_sum, AscendC::LocalTensor<acc_t>& o_scale) {
        // copy co2 -> veccalc
        {
            AscendC::LocalTensor<acc_t> s_co2 = s_co2_que.DeQue<acc_t>();
            AscendC::LocalTensor<acc_t> s = s_que.AllocTensor<acc_t>();
            AscendC::DataCopy(s, s_co2, {1, (uint16_t)(group_size * page_size / 16), 0, 0});
            s_que.EnQue(s);
            s_co2_que.FreeTensor(s_co2);
        }

        // softmax
        {
            AscendC::LocalTensor<acc_t> s = s_que.DeQue<acc_t>();
            AscendC::LocalTensor<acc_t> p_f32 = p_f32_que.AllocTensor<acc_t>();
            /*
            // s: [group_size, page_size]
            for (int i = 0; i < group_size; ++i) {
                // rowmax
                acc_t prev_max = row_max.GetValue(i);
                acc_t curr_max = prev_max;
                for (int j = 0; j < page_size; ++j) {
                    auto val = s.GetValue(i * page_size + j);
                    curr_max = max(curr_max, val);
                }

                // exp
                acc_t scores_scale = exp(prev_max * scale - curr_max);
                o_scale.SetValue(i, scores_scale);
                auto prev_sum = row_sum.GetValue(i);
                row_sum.SetValue(i, prev_sum * scores_scale);
                row_max.SetValue(i, curr_max);

                for (int j = 0; j < page_size; ++j) {
                    auto val = s.GetValue(i * page_size + j);
                    p_f32.SetValue(i * page_size + j, exp(val - curr_max));
                }
            }
            */
            AscendC::DataCopy(p_f32, s, {1, (uint16_t)(group_size * page_size / 16), 0, 0});
            s_que.FreeTensor(s);
            p_f32_que.EnQue(p_f32);
        }

        // p_f32 -> p
        {
            AscendC::LocalTensor<acc_t> p_f32 = p_f32_que.DeQue<acc_t>();
            AscendC::LocalTensor<scalar_t> p = p_que.AllocTensor<scalar_t>();
            Cast(p, p_f32, AscendC::RoundMode::CAST_NONE, group_size * page_size);
            p_que.EnQue(p);
            p_f32_que.FreeTensor(p_f32);
        }
    }


    __aicore__ inline void LoadP() {
        // vecout -> a1
        {
            // nd -> nz
            AscendC::LocalTensor<scalar_t> p = p_que.DeQue<scalar_t>();
            AscendC::LocalTensor<scalar_t> p_a1 = p_a1_que.AllocTensor<scalar_t>();
            for (int i = 0; i < group_size / 16; ++i) {
                int src_offset = i * 16;
                int dst_offset = i * 16 * group_size;
                AscendC::DataCopy(p_a1[dst_offset], p[src_offset], { (uint16_t)group_size, 1, uint16_t(page_size / 16 - 1), 0 });
            }
            p_a1_que.EnQue(p_a1);
            p_que.FreeTensor(p);
        }
        // a1 -> a2
        {
            // nz -> zz
            AscendC::LocalTensor<scalar_t> p_a2 = p_a2_que.AllocTensor<scalar_t>();
            AscendC::LocalTensor<scalar_t> p_a1 = p_a1_que.DeQue<scalar_t>();
            for (int i = 0; i < group_size / 16; ++i) {
                int src_offset = i * 16 * 16;
                int dst_offset = i * 16 * page_size;
                AscendC::LoadData2dParams params;
                params.repeatTimes = page_size / 16;
                params.srcStride = group_size / 16;
                params.ifTranspose = false;
                AscendC::LoadData(p_a2[dst_offset], p_a1[src_offset], params);
            }
            p_a2_que.EnQue(p_a2);
            p_a1_que.FreeTensor(p_a1);
        }
    }

    __aicore__ inline void MmaPV(AscendC::LocalTensor<acc_t>& o, bool zeroed) {
        // TODO
        AscendC::LocalTensor<scalar_t> p_a2 = p_a2_que.DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> v_b2 = v_b2_que.DeQue<scalar_t>();

        AscendC::MmadParams params;
        params.m = BLOCK_M;
        params.n = head_dim;
        params.k = page_size;
        params.cmatrixInitVal = zeroed;
        AscendC::Mmad(o, p_a2, v_b2, params);
    }

    __aicore__ inline void StoreO() {
        // co1 -> co2
        // nz -> nz
        {
            AscendC::LocalTensor<acc_t> o_co1 = o_co1_que.DeQue<acc_t>();
            AscendC::LocalTensor<acc_t> o_co2 = o_co2_que.AllocTensor<acc_t>();
            AscendC::DataCopyParams params;
            params.blockCount = 1;
            params.blockLen = (group_size * head_dim) / (16 * 16);
            AscendC::DataCopyEnhancedParams enhanced_params;
            enhanced_params.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
            AscendC::DataCopy(o_co2, o_co1, params, enhanced_params);
            o_co2_que.EnQue(o_co2);
            o_co1_que.FreeTensor(o_co1);
        }
        // co2 -> gm
        {
            AscendC::LocalTensor<acc_t> o_co2 = o_co2_que.DeQue<acc_t>();
            AscendC::LocalTensor<scalar_t> o_cast = o_que.DeQue<scalar_t>();
            Cast(o_cast, o_co2, AscendC::RoundMode::CAST_NONE, group_size * head_dim);
            o_que.EnQue(o_cast);
            o_co2_que.FreeTensor(o_co2);

            AscendC::LocalTensor<scalar_t> o = o_que.DeQue<scalar_t>();
            // nz -> nd
            for (int i = 0; i < head_dim / 16; ++i) {
                int src_offset = i * group_size * 16;
                int dst_offset = i * 16;
                AscendC::DataCopy(o_gm[dst_offset], o[src_offset], { (uint16_t)group_size, 1, 0, uint16_t(head_dim / 16 - 1)});
            }
            o_que.FreeTensor(o);
        }
    }
};

extern "C" __global__ __aicore__ void paged_attention_ex(GM_ADDR q, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR context_lens, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    using scalar_t = half;
    using acc_t = float;

    // q/y: [bs, num_heads, head_dim]
    // kv_cache: [num_pages, num_kv_heads * head_dim // 16, page_size, 16]
    // block_tables: [bs, ceil(max_seqlen / page_size)]
    // context_lens: [bs]


    uint32_t num_heads = tiling_data.num_heads;
    uint32_t num_kv_heads = tiling_data.num_kv_heads;
    uint32_t head_dim = tiling_data.head_dim;
    uint32_t page_size = tiling_data.page_size;
    uint32_t max_page_num_per_seq = tiling_data.max_page_num_per_seq;
    float scale = tiling_data.scale;
    float scale_log2 = tiling_data.scale_log2;
    int64_t stride_qo_bs = tiling_data.stride_qo_bs;
    int64_t stride_qo_h = tiling_data.stride_qo_h;
    int64_t stride_qo_d = tiling_data.stride_qo_d;
    int64_t stride_kv_p = tiling_data.stride_kv_p;
    int64_t stride_kv_h = tiling_data.stride_kv_h;
    int64_t stride_tables_bs = tiling_data.stride_tables_bs;

    __gm__ scalar_t* q_ptr = reinterpret_cast<__gm__ scalar_t*>(q);
    __gm__ scalar_t* key_cache_ptr = reinterpret_cast<__gm__ scalar_t*>(key_cache);
    __gm__ scalar_t* value_cache_ptr = reinterpret_cast<__gm__ scalar_t*>(value_cache);
    __gm__ int32_t* block_tables_ptr = reinterpret_cast<__gm__ int32_t*>(block_tables);
    __gm__ int32_t* context_lens_ptr = reinterpret_cast<__gm__ int32_t*>(context_lens);
    __gm__ scalar_t* o_ptr = reinterpret_cast<__gm__ scalar_t*>(o);

    PagedAttention<scalar_t, acc_t> paged_attention(num_heads, num_kv_heads, head_dim, page_size, max_page_num_per_seq, scale, scale_log2);
    paged_attention.Init(q_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr, context_lens_ptr, o_ptr, stride_qo_bs, stride_qo_h, stride_qo_d, stride_kv_p, stride_kv_h, stride_tables_bs);
    paged_attention.Process();
}

