
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddRMSNormExTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, num_tokens);
  TILING_DATA_FIELD_DEF(uint32_t, dim);
  TILING_DATA_FIELD_DEF(uint32_t, core_num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddRMSNormEx, AddRMSNormExTilingData)
}

