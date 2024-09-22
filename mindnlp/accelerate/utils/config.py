"""acclerate config"""
from typing import Union, Optional
from dataclasses import dataclass


@dataclass
class MindformersTrainningConfig:
    seed: int = None
    output_dir: str = "./output"
    training_iters: int = 1
    epochs: int = None
    log_interval: int = None
    eval_interval: int = None
    save_interval: int = None
    best_metric_comparison: str = None
    eval_metric: str = None
    grad_clip_kwargs: dict = None
    loss_scale: Union[float, int] = None
    loss_scale_value: Union[float, int] = None
    loss_scale_factor: int = None
    loss_scale_window: int = None
    loss_reduction: str = "mean"
    calculate_per_token_loss: bool = False
    wrap_with_ddp: bool = False
    overlap_grad_reduce: bool = False
    use_distributed_optimizer: bool = False
    bucket_size: Optional[int] = None
    check_for_nan_in_grad: bool = False


@dataclass
class MindForemrsOptimizerConfig:
    optimizer_type: str = "AdamWeightDecay"
    learning_rate: float = 1e-3
    learning_rate_scheduler_kwargs: dict = None
    weight_decay: float = 0.0
    weight_decay_kwargs: dict = None
    zero_config: dict = None


@dataclass
class MindFormersModelParallelConfig:
    tensor_parallel: int = 1
    pipeline_stage: int = 1
    context_parallel: int = 1
    expert_parallel: int = 1
    virtual_pipeline_model_parallel_size: int = None
    micro_batch_num: int = 1
    use_sequence_parallel: bool = False
    recv_dtype: str = "float32"
    zero_level: bool = None
    gradient_accumulation_fusion: bool = False
    standalone_embedding_stage: bool = False
    overlap_p2p_comm: bool = False


@dataclass
class MindFormersDatasetConfig:
    dataset_dir: str = "./dataset"
    shuffle: bool = False
    batch_size: int = 1
    micro_batch_num: int = 1


@dataclass
class MindFormersTransformerConfig:
    vocab_size: int
    num_layers: int
    num_heads: int
    hidden_size: int
    ffn_hidden_size: int
    seq_length: int = None
    attention_type: str = "self_attn"
    position_embedding_type: str = 'absolute'
    parallel_position_embedding: bool = False
    rotary_config: dict = None
    use_query_layer: bool = False
    use_visual_encoder: bool = False
    use_retriever: bool = False
    use_gqa: bool = False
    kv_num_heads: int = 32
    qkv_has_bias: bool = True
    out_proj_has_bias: bool = True
    apply_query_key_layer_scaling: bool = False
    use_flash_attention: bool = False
    fa_config = None
    mask_func_type: str = "attn_mask_add"
    mlp_has_bias: bool = True
    mlp_has_gate: bool = False
    hidden_act: str = "gelu"
    normalization: str = "LayerNorm"
    layernorm_epsilon: float = 1.0e-5
    apply_residual_connection_post_norm: bool = False
    use_final_norm: bool = True
    residual_connection_dtype: str = "float32"
    init_method_std: float = 0.01
    param_init_dtype: str = "float32"
    embedding_init_dtype: str = "float32"
    compute_dtype: str = "float16"
    softmax_compute_dtype: str = "float32"
    init_method: str = 'normal'
    bias_init: str = 'zeros'
    fp16_lm_cross_entropy: bool = False
    hidden_dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    out_hidden_size: int = None
    num_experts: int = None
    untie_embeddings_and_output_weights: bool = False
    flatten_labels_and_input_mask: bool = True
    recompute_method: str = None
    recompute_num_layers: int = None
    recompute_granularity: str = None
