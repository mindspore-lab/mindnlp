
#include "add_rms_norm_ex_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  AddRMSNormExTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::StorageShape* residual_shape = context->GetInputShape(1);
  const gert::StorageShape* weight_shape = context->GetInputShape(2);
  const gert::StorageShape* epsilon_shape = context->GetInputShape(3);
  const gert::StorageShape* y_shape = context->GetOutputShape(0);

  // 从输入shape获取维度信息
  int num_tokens = x1_shape->GetStorageShape().GetDim(0);
  int dim = x1_shape->GetStorageShape().GetDim(1);
  int core_num = (num_tokens < 65535) ? num_tokens : 65535;

  // 设置tiling参数
  tiling.set_num_tokens(num_tokens);
  tiling.set_dim(dim);
  tiling.set_core_num(core_num);

  context->SetBlockDim(core_num);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* residual_shape = context->GetInputShape(1);
    const gert::Shape* weight_shape = context->GetInputShape(2);
    const gert::Shape* epsilon_shape = context->GetInputShape(3);

    // 输出shape与输入x相同
    int num_tokens = x1_shape->GetDim(0);
    int dim = x1_shape->GetDim(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDim(0, num_tokens);
    y_shape->SetDim(1, dim);
    // 新增 residual_output 的 shape 推理
    gert::Shape* residual_output_shape = context->GetOutputShape(1);
    residual_output_shape->SetDim(0, num_tokens);
    residual_output_shape->SetDim(1, dim);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
context->SetOutputDataType(1, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class AddRMSNormEx : public OpDef {
public:
    explicit AddRMSNormEx(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("residual")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("epsilon")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("residual_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");

    }
};

OP_ADD(AddRMSNormEx);
}

