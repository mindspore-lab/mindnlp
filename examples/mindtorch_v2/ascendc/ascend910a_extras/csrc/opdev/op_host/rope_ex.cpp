
#include "rope_ex_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  RopeExTilingData tiling;
  const gert::StorageShape* q_shape = context->GetInputShape(0);
  const gert::StorageShape* k_shape = context->GetInputShape(1);
  // q: [bs, num_heads, head_dim]
  // k: [bs, num_kv_heads, head_dim]
  int bs = q_shape->GetStorageShape().GetDim(0);
  int num_heads = q_shape->GetStorageShape().GetDim(1);
  int head_dim = q_shape->GetStorageShape().GetDim(2);
  int num_kv_heads = k_shape->GetStorageShape().GetDim(1);
  tiling.set_num_heads(num_heads);
  tiling.set_num_kv_heads(num_kv_heads);
  tiling.set_head_dim(head_dim);
  context->SetBlockDim(bs);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  size_t *workspace_sizes = context->GetWorkspaceSizes(1);
  workspace_sizes[0] = 0;

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* q_shape = context->GetInputShape(0);
    const gert::Shape* k_shape = context->GetInputShape(1);
    gert::Shape* out_q_shape = context->GetOutputShape(0);
    gert::Shape* out_k_shape = context->GetOutputShape(1);
    *out_q_shape = *q_shape;
    *out_k_shape = *k_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_dtype);
    context->SetOutputDataType(1, input_dtype);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class RopeEx : public OpDef {
public:
    explicit RopeEx(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("position_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cos_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sin_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(RopeEx);
}

