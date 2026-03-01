
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReshapeAndCacheExTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, num_tokens);
  TILING_DATA_FIELD_DEF(uint32_t, num_kv_heads);
  TILING_DATA_FIELD_DEF(uint32_t, head_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_num);
  TILING_DATA_FIELD_DEF(uint32_t, num_blocks);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, nh16);
  TILING_DATA_FIELD_DEF(uint32_t, h16);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReshapeAndCacheEx, ReshapeAndCacheExTilingData)
}

