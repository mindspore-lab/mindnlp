
#include "swi_glu_ex_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  SwiGluExTilingData tiling;
  const gert::StorageShape* x_shape = context->GetInputShape(0);
  int num_tokens = x_shape->GetStorageShape().GetDim(0);
  int dim = x_shape->GetStorageShape().GetDim(1) / 2;
  int core_num = (num_tokens < 65535) ? num_tokens : 65535;

  context->SetBlockDim(core_num);
  tiling.set_num_tokens(num_tokens);
  tiling.set_dim(dim);
  tiling.set_core_num(core_num);
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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    y_shape->SetDim(1, x1_shape->GetDim(1) / 2);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SwiGluEx : public OpDef {
public:
    explicit SwiGluEx(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910");

    }
};

OP_ADD(SwiGluEx);
}

