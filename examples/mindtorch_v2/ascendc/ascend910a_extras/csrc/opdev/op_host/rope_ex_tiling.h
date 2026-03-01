
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RopeExTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, num_heads);
  TILING_DATA_FIELD_DEF(uint32_t, num_kv_heads);
  TILING_DATA_FIELD_DEF(uint32_t, head_dim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RopeEx, RopeExTilingData)
}

