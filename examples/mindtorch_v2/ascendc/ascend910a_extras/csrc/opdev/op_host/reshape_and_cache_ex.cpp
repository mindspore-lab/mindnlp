
#include "reshape_and_cache_ex_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ReshapeAndCacheExTilingData tiling;
  const gert::StorageShape* key_shape = context->GetInputShape(0);
  const gert::StorageShape* value_shape = context->GetInputShape(1);
  const gert::StorageShape* key_cache_shape = context->GetInputShape(2);
  const gert::StorageShape* value_cache_shape = context->GetInputShape(3);
  const gert::StorageShape* slot_indices_shape = context->GetInputShape(4);

  int32_t num_tokens = key_shape->GetStorageShape().GetDim(0);
  int32_t num_kv_heads = key_shape->GetStorageShape().GetDim(1);
  int32_t head_size = key_shape->GetStorageShape().GetDim(2);
  int32_t core_num = (num_tokens < 65535) ? num_tokens : 65535;
  int32_t num_blocks = key_cache_shape->GetStorageShape().GetDim(0);
  int32_t block_size = key_cache_shape->GetStorageShape().GetDim(1);
  int32_t nh16 = key_cache_shape->GetStorageShape().GetDim(2);
  int32_t h16 = key_cache_shape->GetStorageShape().GetDim(3);

  int key_buffer_size = num_tokens * num_kv_heads * head_size;
  int key_cache_buffer_size = num_blocks * block_size * nh16 * h16;
  int value_buffer_size = num_tokens * num_kv_heads * head_size;
  int value_cache_buffer_size = num_blocks * block_size * nh16 * h16;

  int32_t slot_indices_len = slot_indices_shape->GetStorageShape().GetDim(0);
  // host bound check: slot_indices length must be equal to num_tokens, otherwise return failed, prevent overflow
  if (slot_indices_len != num_tokens) {
    // slot_indices length is invalid, prevent kernel overflow
    return ge::GRAPH_FAILED;
  }

  tiling.set_num_tokens(num_tokens);
  tiling.set_num_kv_heads(num_kv_heads);
  tiling.set_head_size(head_size);
  tiling.set_core_num(core_num);
  tiling.set_num_blocks(num_blocks);
  tiling.set_block_size(block_size);
  tiling.set_nh16(nh16);
  tiling.set_h16(h16);
  context->SetBlockDim(core_num);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* key_shape = context->GetInputShape(0);
    const gert::Shape* value_shape = context->GetInputShape(1);
    const gert::Shape* key_cache_shape = context->GetInputShape(2);
    const gert::Shape* value_cache_shape = context->GetInputShape(3);
    const gert::Shape* slot_indices_shape = context->GetInputShape(4);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class ReshapeAndCacheEx : public OpDef {
public:
    explicit ReshapeAndCacheEx(const char* name) : OpDef(name)
    {
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value_cache")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("slot_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");

    }
};

OP_ADD(ReshapeAndCacheEx);
}

