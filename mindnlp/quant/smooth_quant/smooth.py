'''
code from https://github.com/mit-han-lab/smoothquant/
'''
from mindnlp.core import ops, nn, no_grad

from mindnlp.transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

@no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (LlamaRMSNorm, nn.Linear))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.shape[0] == fc.in_features == act_scales.numel()
    dtype = fcs[0].weight.dtype
    act_scales = act_scales.to(dtype=dtype)
    weight_scales = ops.cat(
        [ops.max(fc.weight.abs(), dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = ops.max(weight_scales, dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(dtype)
    )
    if ln.weight.dim() == 2:
        ln.weight = ln.weight.div(scales.unsqueeze(-1))
    else:
        ln.weight = ln.weightdiv(scales)
    for fc in fcs:
        fc.weight = fc.weight.mul(scales.view(1, -1))


@no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
