
#include <cmath>
#include "paged_attention_ex_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  PagedAttentionExTilingData tiling;
  const gert::StorageShape* q_shape = context->GetInputShape(0);
  const gert::StorageShape* key_cache_shape = context->GetInputShape(1);
  const gert::StorageShape* value_cache_shape = context->GetInputShape(2);
  const gert::StorageShape* block_tables_shape = context->GetInputShape(3);
  const gert::StorageShape* context_lens_shape = context->GetInputShape(4);

  // q: [bs, num_heads, head_dim]
  // kvcache: [num_pages, num_kv_heads * head_dim // 16, page_size, 16]
  int num_heads = q_shape->GetStorageShape().GetDim(1);
  int head_dim = q_shape->GetStorageShape().GetDim(2);
  int num_kv_heads = key_cache_shape->GetStorageShape().GetDim(1) * 16 / head_dim;
  int num_pages = block_tables_shape->GetStorageShape().GetDim(0);
  int max_page_num_per_seq = block_tables_shape->GetStorageShape().GetDim(1);
  int page_size = key_cache_shape->GetStorageShape().GetDim(2);
  float scale = 1.0f / std::sqrt(head_dim);
  float scale_log2 = scale * 1.44269504088896340736f;

  int64_t stride_qo_bs = num_heads * head_dim;
  int64_t stride_qo_h = head_dim;
  int64_t stride_qo_d = 1;
  int64_t stride_kv_p = num_kv_heads * head_dim * page_size;
  int64_t stride_kv_h = head_dim * page_size;
  int64_t stride_tables_bs = max_page_num_per_seq;

  int group_size = num_heads / num_kv_heads;
  int bs = q_shape->GetStorageShape().GetDim(0);

  tiling.set_num_heads(num_heads);
  tiling.set_num_kv_heads(num_kv_heads);
  tiling.set_head_dim(head_dim);
  tiling.set_num_pages(num_pages);
  tiling.set_page_size(page_size);
  tiling.set_max_page_num_per_seq(max_page_num_per_seq);
  tiling.set_scale(scale);
  tiling.set_scale_log2(scale_log2);
  tiling.set_stride_qo_bs(stride_qo_bs);
  tiling.set_stride_qo_h(stride_qo_h);
  tiling.set_stride_qo_d(stride_qo_d);
  tiling.set_stride_kv_p(stride_kv_p);
  tiling.set_stride_kv_h(stride_kv_h);
  tiling.set_stride_tables_bs(stride_tables_bs);
  printf("attn: bs=%d, num_heads=%d, num_kv_heads=%d, group_size=%d, head_dim=%d, num_pages=%d, page_size=%d, max_page_num_per_seq=%d, scale=%f\n", bs, num_heads, num_kv_heads, group_size, head_dim, num_pages, page_size, max_page_num_per_seq, scale);
  context->SetBlockDim(bs * num_kv_heads);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* q_shape = context->GetInputShape(0);
    gert::Shape* o_shape = context->GetOutputShape(0);
    *o_shape = *q_shape;
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
class PagedAttentionEx : public OpDef {
public:
    explicit PagedAttentionEx(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("block_tables")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("context_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("o")
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

OP_ADD(PagedAttentionEx);
}

