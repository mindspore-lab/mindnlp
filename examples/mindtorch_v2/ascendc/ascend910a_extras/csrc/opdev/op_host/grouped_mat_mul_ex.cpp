
#include "grouped_mat_mul_ex_tiling.h"
#include "register/op_def_registry.h"


#include <cstdio>
#include <cassert>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  GroupedMatMulExTilingData tiling;
  const gert::StorageShape* x_shape = context->GetInputShape(0);
  const gert::StorageShape* w_shape = context->GetInputShape(1);
  int num_tokens = x_shape->GetStorageShape().GetDim(0);
  int dim = x_shape->GetStorageShape().GetDim(1);
  int num_exports = w_shape->GetStorageShape().GetDim(0);
  int inner_dim = w_shape->GetStorageShape().GetDim(1);
  int core_num = (num_exports < 65535) ? num_exports : 65535;
  // printf("num_tokens: %d, dim: %d, num_exports: %d, inner_dim: %d, core_num: %d\n", num_tokens, dim, num_exports, inner_dim, core_num);
  context->SetBlockDim(core_num);
  tiling.set_num_tokens(num_tokens);
  tiling.set_dim(dim);
  tiling.set_num_exports(num_exports);
  tiling.set_inner_dim(inner_dim);
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
    const gert::Shape* x_shape = context->GetInputShape(0);
    const gert::Shape* w_shape = context->GetInputShape(1);
    int num_tokens = x_shape->GetDim(0);
    int dim = x_shape->GetDim(1);
    int num_exports = w_shape->GetDim(0);
    int inner_dim = w_shape->GetDim(1);
    if (w_shape->GetDim(2) != dim) {
        return GRAPH_FAILED;
    }
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDim(0, num_tokens);
    y_shape->SetDim(1, inner_dim);
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
class GroupedMatMulEx : public OpDef {
public:
    explicit GroupedMatMulEx(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("group_list")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
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
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(GroupedMatMulEx);
}

