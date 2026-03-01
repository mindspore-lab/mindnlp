
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PagedAttentionExTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, num_heads);
  TILING_DATA_FIELD_DEF(uint32_t, num_kv_heads);
  TILING_DATA_FIELD_DEF(uint32_t, head_dim);
  TILING_DATA_FIELD_DEF(uint32_t, num_pages);
  TILING_DATA_FIELD_DEF(uint32_t, page_size);
  TILING_DATA_FIELD_DEF(uint32_t, max_page_num_per_seq);
  TILING_DATA_FIELD_DEF(float, scale);
  TILING_DATA_FIELD_DEF(float, scale_log2);
  TILING_DATA_FIELD_DEF(int64_t, stride_qo_bs);
  TILING_DATA_FIELD_DEF(int64_t, stride_qo_h);
  TILING_DATA_FIELD_DEF(int64_t, stride_qo_d);
  TILING_DATA_FIELD_DEF(int64_t, stride_kv_p);
  TILING_DATA_FIELD_DEF(int64_t, stride_kv_h);
  TILING_DATA_FIELD_DEF(int64_t, stride_tables_bs);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PagedAttentionEx, PagedAttentionExTilingData)
}

