#include "kernel_operator.h"

extern "C" __global__ __aicore__ void reshape_and_cache_ex(
    GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR slot_indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    int num_tokens = tiling_data.num_tokens;
    int num_kv_heads = tiling_data.num_kv_heads;
    int head_size = tiling_data.head_size;
    int core_num = tiling_data.core_num;
    int num_blocks = tiling_data.num_blocks;
    int block_size = tiling_data.block_size;
    int nh16 = tiling_data.nh16;
    int h16 = tiling_data.h16;

    using scalar_t = half;

    __gm__ scalar_t *key_ptr = reinterpret_cast<__gm__ scalar_t *>(key);
    __gm__ scalar_t *value_ptr = reinterpret_cast<__gm__ scalar_t *>(value);
    __gm__ scalar_t *key_cache_ptr = reinterpret_cast<__gm__ scalar_t *>(key_cache);
    __gm__ scalar_t *value_cache_ptr = reinterpret_cast<__gm__ scalar_t *>(value_cache);
    __gm__ int32_t *slot_indices_ptr = reinterpret_cast<__gm__ int32_t *>(slot_indices);

    // check value and value_cache is nullptr (optional input)
    bool has_value = (value_ptr != nullptr);
    bool has_value_cache = (value_cache_ptr != nullptr);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> key_que;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> value_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> key_cache_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> value_cache_que;

    // increase buffer size to 16, match reference implementation
    constexpr int BLOCK_SIZE = 16;
    pipe.InitBuffer(key_que, 1, sizeof(scalar_t) * BLOCK_SIZE);
    pipe.InitBuffer(key_cache_que, 1, sizeof(scalar_t) * BLOCK_SIZE);

    // only init related queues when value and value_cache exist
    if (has_value && has_value_cache) {
        pipe.InitBuffer(value_que, 1, sizeof(scalar_t) * BLOCK_SIZE);
        pipe.InitBuffer(value_cache_que, 1, sizeof(scalar_t) * BLOCK_SIZE);
    }

    int key_buffer_size = num_tokens * num_kv_heads * head_size;
    int key_cache_buffer_size = num_blocks * block_size * nh16 * 16;
    // transform key to key_cache
    for (int64_t i = 0; i < num_tokens; ++i) {
        int32_t slot = slot_indices_ptr[i];
        // kernel Bound check: slot must be in the valid range
        if (slot < 0 || slot >= nh16 * num_blocks) continue;
        int block = slot / nh16; // nh16 = 128
        int nh16_idx = slot % nh16;
        int idx = 0;
        for (int block_offset = 0; block_offset < block_size; ++block_offset) {
            for (int j = 0; j < 16; ++j) {
                if (idx < num_kv_heads * head_size) {
                    int key_src_offset = i * num_kv_heads * head_size + idx;
                    int cache_dst_offset = ((block * block_size + block_offset) * nh16 + nh16_idx) * 16 + j;
                    // bound check: offset must be in the valid range
                    if (key_src_offset < key_buffer_size && cache_dst_offset < key_cache_buffer_size) {
                        key_cache_ptr[cache_dst_offset] = key_ptr[key_src_offset];
                        if (has_value && has_value_cache) {
                            value_cache_ptr[cache_dst_offset] = value_ptr[key_src_offset];
                        }
                    }
                    idx++;
                }
            }
        }
    }
}

