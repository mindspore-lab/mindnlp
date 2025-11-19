from mindspore.ops.auto_generate.gen_ops_prim import *
from mindspore.ops.auto_generate.pyboost_inner_prim import *

abs_op = Abs().set_device('CPU')

acos_ext_op = AcosExt().set_device('CPU')

acosh_ext_op = AcoshExt().set_device('CPU')

adamw_op = AdamW().set_device('CPU')

adaptive_avg_pool1d_op = AdaptiveAvgPool1D().set_device('CPU')

adaptive_avg_pool2d_ext_op = AdaptiveAvgPool2DExt().set_device('CPU')

adaptive_avg_pool2d_grad_ext_op = AdaptiveAvgPool2DGradExt().set_device('CPU')

adaptive_avg_pool3d_ext_op = AdaptiveAvgPool3DExt().set_device('CPU')

adaptive_avg_pool3d_grad_ext_op = AdaptiveAvgPool3DGradExt().set_device('CPU')

adaptive_max_pool1d_op = AdaptiveMaxPool1D().set_device('CPU')

add_op = Add().set_device('CPU')

add_ext_op = AddExt().set_device('CPU')

add_layer_norm_grad_op = AddLayerNormGrad().set_device('CPU')

add_layernorm_v2_op = AddLayerNormV2().set_device('CPU')

add_rms_norm_op = AddRmsNorm().set_device('CPU')

add_scalar_op = AddScalar().set_device('CPU')

addbmm_op = Addbmm().set_device('CPU')

addcdiv_ext_op = AddcdivExt().set_device('CPU')

addcmul_ext_op = AddcmulExt().set_device('CPU')

addmm_op = Addmm().set_device('CPU')

addmv_op = Addmv().set_device('CPU')

all_gather_matmul_op = AllGatherMatmul().set_device('CPU')

arange_op = Arange().set_device('CPU')

argmax_ext_op = ArgMaxExt().set_device('CPU')

argmin_ext_op = ArgMinExt().set_device('CPU')

argsort_op = ArgSort().set_device('CPU')

as_strided_op = AsStrided().set_device('CPU')

asin_ext_op = AsinExt().set_device('CPU')

asinh_ext_op = AsinhExt().set_device('CPU')

atan2_ext_op = Atan2Ext().set_device('CPU')

atan_ext_op = AtanExt().set_device('CPU')

atanh_op = Atanh().set_device('CPU')

avg_pool1d_op = AvgPool1D().set_device('CPU')

avg_pool2d_op = AvgPool2D().set_device('CPU')

avg_pool2d_grad_op = AvgPool2DGrad().set_device('CPU')

avg_pool3d_ext_op = AvgPool3DExt().set_device('CPU')

avg_pool3d_grad_ext_op = AvgPool3DGradExt().set_device('CPU')

baddbmm_op = Baddbmm().set_device('CPU')

batch_norm_elemt_op = BatchNormElemt().set_device('CPU')

batch_norm_elemt_grad_op = BatchNormElemtGrad().set_device('CPU')

batch_norm_ext_op = BatchNormExt().set_device('CPU')

batch_norm_gather_stats_with_counts_op = BatchNormGatherStatsWithCounts().set_device('CPU')

batch_norm_reduce_grad_op = BatchNormReduceGrad().set_device('CPU')

batch_norm_stats_op = BatchNormStats().set_device('CPU')

bernoulli_ext_op = BernoulliExt().set_device('CPU')

binary_cross_entropy_with_logits_backward_op = BinaryCrossEntropyWithLogitsBackward().set_device('CPU')

bincount_ext_op = BincountExt().set_device('CPU')

bitwise_and_scalar_op = BitwiseAndScalar().set_device('CPU')

bitwise_and_tensor_op = BitwiseAndTensor().set_device('CPU')

bitwise_not_op = BitwiseNot().set_device('CPU')

bitwise_or_scalar_op = BitwiseOrScalar().set_device('CPU')

bitwise_or_tensor_op = BitwiseOrTensor().set_device('CPU')

bitwise_xor_scalar_op = BitwiseXorScalar().set_device('CPU')

bitwise_xor_tensor_op = BitwiseXorTensor().set_device('CPU')

bmm_ext_op = BatchMatMulExt().set_device('CPU')

broadcast_to_view_op = BroadcastToView().set_device('CPU')

ceil_op = Ceil().set_device('CPU')

chunk_op = Chunk().set_device('CPU')

chunk_view_op = ChunkView().set_device('CPU')

clamp_scalar_op = ClampScalar().set_device('CPU')

clamp_tensor_op = ClampTensor().set_device('CPU')

clone_op = Clone().set_device('CPU')

col2im_ext_op = Col2ImExt().set_device('CPU')

col2im_grad_op = Col2ImGrad().set_device('CPU')

constant_pad_nd_op = ConstantPadND().set_device('CPU')

contiguous_op = Contiguous().set_device('CPU')

conv1d_ext_op = Conv1DExt().set_device('CPU')

conv1d_padding_op = Conv1DPadding().set_device('CPU')

conv2d_ext_op = Conv2DExt().set_device('CPU')

conv2d_padding_op = Conv2DPadding().set_device('CPU')

conv3d_ext_op = Conv3DExt().set_device('CPU')

conv3d_padding_op = Conv3DPadding().set_device('CPU')

conv_transpose2d_op = ConvTranspose2D().set_device('CPU')

convolution_op = Convolution().set_device('CPU')

convolution_grad_op = ConvolutionGrad().set_device('CPU')

convolution_str_op = ConvolutionStr().set_device('CPU')

convolution_str_grad_op = ConvolutionStrGrad().set_device('CPU')

copy_op = Copy().set_device('CPU')

cos_op = Cos().set_device('CPU')

cosh_op = Cosh().set_device('CPU')

count_nonzero_op = CountNonZero().set_device('CPU')

cummin_ext_op = CumminExt().set_device('CPU')

cumsum_ext_op = CumsumExt().set_device('CPU')

dense_op = Dense().set_device('CPU')

diag_ext_op = DiagExt().set_device('CPU')

dist_comm_all_gather_op = DistCommAllGather().set_device('CPU')

dist_comm_all_gather_into_tensor_op = DistCommAllGatherIntoTensor().set_device('CPU')

dist_comm_all_reduce_op = DistCommAllReduce().set_device('CPU')

dist_comm_all_to_all_v_op = DistCommAllToAllV().set_device('CPU')

dist_comm_all_to_all_v_single_op = DistCommAllToAllVSingle().set_device('CPU')

dist_comm_barrier_op = DistCommBarrier().set_device('CPU')

dist_comm_batch_isend_irecv_op = DistCommBatchIsendIrecv().set_device('CPU')

dist_comm_broadcast_op = DistCommBroadcast().set_device('CPU')

dist_comm_gather_op = DistCommGather().set_device('CPU')

dist_comm_gather_into_tensor_op = DistCommGatherIntoTensor().set_device('CPU')

dist_comm_irecv_op = DistCommIrecv().set_device('CPU')

dist_comm_isend_op = DistCommIsend().set_device('CPU')

dist_comm_reduce_op = DistCommReduce().set_device('CPU')

dist_comm_reduce_scatter_op = DistCommReduceScatter().set_device('CPU')

dist_comm_reduce_scatter_tensor_op = DistCommReduceScatterTensor().set_device('CPU')

dist_comm_scatter_op = DistCommScatter().set_device('CPU')

dist_comm_scatter_tensor_op = DistCommScatterTensor().set_device('CPU')

div_op = Div().set_device('CPU')

divmod_op = DivMod().set_device('CPU')

divmods_op = DivMods().set_device('CPU')

divs_op = Divs().set_device('CPU')

dot_op = Dot().set_device('CPU')

dropout_do_mask_ext_op = DropoutDoMaskExt().set_device('CPU')

dropout_ext_op = DropoutExt().set_device('CPU')

dropout_gen_mask_ext_op = DropoutGenMaskExt().set_device('CPU')

dropout_grad_ext_op = DropoutGradExt().set_device('CPU')

dynamic_quant_ext_op = DynamicQuantExt().set_device('CPU')

elu_grad_ext_op = EluGradExt().set_device('CPU')

embedding_op = Embedding().set_device('CPU')

embedding_dense_backward_op = EmbeddingDenseBackward().set_device('CPU')

equal_op = Equal().set_device('CPU')

equal_ext_op = EqualExt().set_device('CPU')

erf_op = Erf().set_device('CPU')

erfc_op = Erfc().set_device('CPU')

erfinv_op = Erfinv().set_device('CPU')

exp_op = Exp().set_device('CPU')

exp2_op = Exp2().set_device('CPU')

expand_as_op = ExpandAs().set_device('CPU')

expand_dims_op = ExpandDims().set_device('CPU')

expand_dims_view_op = ExpandDimsView().set_device('CPU')

expm1_op = Expm1().set_device('CPU')

eye_op = Eye().set_device('CPU')

fill_scalar_op = FillScalar().set_device('CPU')

fill_tensor_op = FillTensor().set_device('CPU')

flatten_ext_op = FlattenExt().set_device('CPU')

floor_op = Floor().set_device('CPU')

floor_div_op = FloorDiv().set_device('CPU')

floor_div_scalar_op = FloorDivScalar().set_device('CPU')

fmod_scalar_op = FmodScalar().set_device('CPU')

fmod_tensor_op = FmodTensor().set_device('CPU')

frac_op = Frac().set_device('CPU')

full_like_op = FullLike().set_device('CPU')

gather_d_op = GatherD().set_device('CPU')

gather_d_grad_v2_op = GatherDGradV2().set_device('CPU')

gcd_op = Gcd().set_device('CPU')

gelu_op = GeLU().set_device('CPU')

gelu_ext_op = GeluExt().set_device('CPU')

gelu_grad_op = GeLUGrad().set_device('CPU')

gelu_grad_ext_op = GeluGradExt().set_device('CPU')

generator_op = Generator().set_device('CPU')

gmm_op = Gmm().set_device('CPU')

gmm_backward_op = GmmBackward().set_device('CPU')

gmm_backward_fusion_op = GmmBackwardFusion().set_device('CPU')

gmm_v2_op = GmmV2().set_device('CPU')

gmm_v2_backward_op = GmmV2Backward().set_device('CPU')

gmm_v2_backward_fusion_op = GmmV2BackwardFusion().set_device('CPU')

greater_op = Greater().set_device('CPU')

greater_equal_op = GreaterEqual().set_device('CPU')

greater_equal_scalar_op = GreaterEqualScalar().set_device('CPU')

group_norm_op = GroupNorm().set_device('CPU')

group_norm_grad_op = GroupNormGrad().set_device('CPU')

grouped_matmul_v2_op = GroupedMatmulV2().set_device('CPU')

grouped_matmul_v4_op = GroupedMatmulV4().set_device('CPU')

hardtanh_op = Hardtanh().set_device('CPU')

hardtanh_grad_op = HardtanhGrad().set_device('CPU')

histc_ext_op = HistcExt().set_device('CPU')

hsigmoid_op = HSigmoid().set_device('CPU')

hsigmoid_grad_op = HSigmoidGrad().set_device('CPU')

hswish_op = HSwish().set_device('CPU')

hswish_grad_op = HSwishGrad().set_device('CPU')

im2col_ext_op = Im2ColExt().set_device('CPU')

index_op = Index().set_device('CPU')

index_add_ext_op = IndexAddExt().set_device('CPU')

index_fill_scalar_op = IndexFillScalar().set_device('CPU')

index_fill_tensor_op = IndexFillTensor().set_device('CPU')

index_select_op = IndexSelect().set_device('CPU')

inner_comm_all_gather_op = InnerCommAllGather().set_device('CPU')

inner_comm_all_reduce_op = InnerCommAllReduce().set_device('CPU')

inner_comm_all_to_all_v_op = InnerCommAllToAllV().set_device('CPU')

inner_comm_irecv_op = InnerCommIrecv().set_device('CPU')

inner_comm_isend_op = InnerCommIsend().set_device('CPU')

inner_comm_reduce_scatter_op = InnerCommReduceScatter().set_device('CPU')

inner_index_op = InnerIndex().set_device('CPU')

inner_inplace_index_put_op = InnerInplaceIndexPut().set_device('CPU')

inner_non_zero_op = InnerNonZero().set_device('CPU')

inplace_add_ext_op = InplaceAddExt().set_device('CPU')

inplace_addmm_op = InplaceAddmm().set_device('CPU')

inplace_adds_ext_op = InplaceAddsExt().set_device('CPU')

inplace_clamp_scalar_op = InplaceClampScalar().set_device('CPU')

inplace_clamp_tensor_op = InplaceClampTensor().set_device('CPU')

inplace_copy_op = InplaceCopy().set_device('CPU')

inplace_div_op = InplaceDiv().set_device('CPU')

inplace_divmod_op = InplaceDivMod().set_device('CPU')

inplace_divmods_op = InplaceDivMods().set_device('CPU')

inplace_divs_op = InplaceDivs().set_device('CPU')

inplace_elu_op = InplaceElu().set_device('CPU')

inplace_erfinv_op = InplaceErfinv().set_device('CPU')

inplace_exp_op = InplaceExp().set_device('CPU')

inplace_exponential_op = InplaceExponential().set_device('CPU')

inplace_fill_diagonal_op = InplaceFillDiagonal().set_device('CPU')

inplace_fill_scalar_op = InplaceFillScalar().set_device('CPU')

inplace_fill_tensor_op = InplaceFillTensor().set_device('CPU')

inplace_floor_op = InplaceFloor().set_device('CPU')

inplace_floor_divide_op = InplaceFloorDivide().set_device('CPU')

inplace_floor_divides_op = InplaceFloorDivides().set_device('CPU')

inplace_grouped_matmul_add_op = InplaceGroupedMatmulAdd().set_device('CPU')

inplace_hardtanh_op = InplaceHardtanh().set_device('CPU')

inplace_index_add_op = InplaceIndexAddExt().set_device('CPU')

inplace_index_put_op = InplaceIndexPut().set_device('CPU')

inplace_log_op = InplaceLog().set_device('CPU')

inplace_masked_fill_scalar_op = InplaceMaskedFillScalar().set_device('CPU')

inplace_masked_fill_tensor_op = InplaceMaskedFillTensor().set_device('CPU')

inplace_mul_op = InplaceMul().set_device('CPU')

inplace_muls_op = InplaceMuls().set_device('CPU')

inplace_normal_op = InplaceNormal().set_device('CPU')

inplace_put_op = InplacePut().set_device('CPU')

inplace_random_op = InplaceRandom().set_device('CPU')

inplace_relu_op = InplaceReLU().set_device('CPU')

inplace_scatter_add_op = InplaceScatterAdd().set_device('CPU')

inplace_scatter_src_op = InplaceScatterSrc().set_device('CPU')

inplace_scatter_src_reduce_op = InplaceScatterSrcReduce().set_device('CPU')

inplace_scatter_value_op = InplaceScatterValue().set_device('CPU')

inplace_scatter_value_reduce_op = InplaceScatterValueReduce().set_device('CPU')

inplace_stop_gradient_op = InplaceStopGradient().set_device('CPU')

inplace_sub_ext_op = InplaceSubExt().set_device('CPU')

inplace_sub_scalar_op = InplaceSubScalar().set_device('CPU')

inplace_tanh_op = InplaceTanh().set_device('CPU')

inplace_threshold_op = InplaceThreshold().set_device('CPU')

inplace_uniform_op = InplaceUniform().set_device('CPU')

inplace_zero_op = InplaceZero().set_device('CPU')

isfinite_op = IsFinite().set_device('CPU')

isinf_op = IsInf().set_device('CPU')

isneginf_op = IsNegInf().set_device('CPU')

kl_div_op = KLDiv().set_device('CPU')

kl_div_grad_op = KLDivGrad().set_device('CPU')

kthvalue_op = Kthvalue().set_device('CPU')

kv_cache_scatter_update_op = KVCacheScatterUpdate().set_device('CPU')

l1_loss_backward_ext_op = L1LossBackwardExt().set_device('CPU')

l1_loss_ext_op = L1LossExt().set_device('CPU')

layer_norm_ext_op = LayerNormExt().set_device('CPU')

layer_norm_grad_ext_op = LayerNormGradExt().set_device('CPU')

leaky_relu_ext_op = LeakyReLUExt().set_device('CPU')

leaky_relu_grad_ext_op = LeakyReLUGradExt().set_device('CPU')

lerp_op = Lerp().set_device('CPU')

lerp_scalar_op = LerpScalar().set_device('CPU')

less_op = Less().set_device('CPU')

less_equal_op = LessEqual().set_device('CPU')

lin_space_ext_op = LinSpaceExt().set_device('CPU')

linalg_qr_op = LinalgQr().set_device('CPU')

linalg_vector_norm_op = LinalgVectorNorm().set_device('CPU')

log_op = Log().set_device('CPU')

log10_op = Log10().set_device('CPU')

log1p_op = Log1p().set_device('CPU')

log2_op = Log2().set_device('CPU')

log_softmax_ext_op = LogSoftmaxExt().set_device('CPU')

logaddexp_op = LogAddExp().set_device('CPU')

logaddexp2_op = LogAddExp2().set_device('CPU')

logical_and_op = LogicalAnd().set_device('CPU')

logical_not_op = LogicalNot().set_device('CPU')

logical_or_op = LogicalOr().set_device('CPU')

logical_xor_op = LogicalXor().set_device('CPU')

logsigmoid_op = LogSigmoid().set_device('CPU')

logsigmoid_grad_op = LogSigmoidGrad().set_device('CPU')

logsumexp_op = LogSumExp().set_device('CPU')

masked_fill_op = MaskedFill().set_device('CPU')

masked_select_op = MaskedSelect().set_device('CPU')

masked_select_grad_op = MaskedSelectGrad().set_device('CPU')

matmul_allreduce_add_rmsnorm_op = MatmulAllReduceAddRmsNorm().set_device('CPU')

matmul_ext_op = MatMulExt().set_device('CPU')

matmul_reduce_scatter_op = MatmulReduceScatter().set_device('CPU')

matrix_inverse_ext_op = MatrixInverseExt().set_device('CPU')

max_op = Max().set_device('CPU')

max_dim_op = MaxDim().set_device('CPU')

max_unpool2d_ext_op = MaxUnpool2DExt().set_device('CPU')

maximum_op = Maximum().set_device('CPU')

mean_ext_op = MeanExt().set_device('CPU')

median_dim_op = MedianDim().set_device('CPU')

median_ext_op = MedianExt().set_device('CPU')

min_op = Min().set_device('CPU')

min_dim_op = MinDim().set_device('CPU')

minimum_op = Minimum().set_device('CPU')

mish_ext_op = MishExt().set_device('CPU')

mish_grad_ext_op = MishGradExt().set_device('CPU')

mm_ext_op = Mm().set_device('CPU')

moe_compute_expert_tokens_op = MoeComputeExpertTokens().set_device('CPU')

moe_finalize_routing_op = MoeFinalizeRouting().set_device('CPU')

moe_gating_top_k_softmax_op = MoeGatingTopKSoftmax().set_device('CPU')

moe_init_routing_op = MoeInitRouting().set_device('CPU')

moe_init_routing_v2_op = MoeInitRoutingV2().set_device('CPU')

moe_token_permute_op = MoeTokenPermute().set_device('CPU')

moe_token_permute_grad_op = MoeTokenPermuteGrad().set_device('CPU')

moe_token_unpermute_op = MoeTokenUnpermute().set_device('CPU')

moe_token_unpermute_grad_op = MoeTokenUnpermuteGrad().set_device('CPU')

mse_loss_ext_op = MSELossExt().set_device('CPU')

mse_loss_grad_ext_op = MSELossGradExt().set_device('CPU')

mul_op = Mul().set_device('CPU')

muls_op = Muls().set_device('CPU')

multi_scale_deformable_attn_op = MultiScaleDeformableAttn().set_device('CPU')

multi_scale_deformable_attn_grad_op = MultiScaleDeformableAttnGrad().set_device('CPU')

multinomial_ext_op = MultinomialExt().set_device('CPU')

mv_op = Mv().set_device('CPU')

nansum_op = Nansum().set_device('CPU')

narrow_op = Narrow().set_device('CPU')

narrow_view_op = NarrowView().set_device('CPU')

neg_op = Neg().set_device('CPU')

new_ones_op = NewOnes().set_device('CPU')

new_zeros_op = NewZeros().set_device('CPU')

nllloss_2d_op = NLLLoss2d().set_device('CPU')

nllloss_2d_grad_op = NLLLoss2dGrad().set_device('CPU')

non_zero_op = NonZero().set_device('CPU')

non_zero_ext_op = NonZeroExt().set_device('CPU')

norm_op = Norm().set_device('CPU')

normal_float_float_op = NormalFloatFloat().set_device('CPU')

normal_float_tensor_op = NormalFloatTensor().set_device('CPU')

normal_tensor_float_op = NormalTensorFloat().set_device('CPU')

normal_tensor_tensor_op = NormalTensorTensor().set_device('CPU')

not_equal_op = NotEqual().set_device('CPU')

ones_like_ext_op = OnesLikeExt().set_device('CPU')

outer_op = Outer().set_device('CPU')

pixel_shuffle_op = PixelShuffle().set_device('CPU')

polar_op = Polar().set_device('CPU')

pow_op = Pow().set_device('CPU')

pow_scalar_tensor_op = PowScalarTensor().set_device('CPU')

pow_tensor_scalar_op = PowTensorScalar().set_device('CPU')

prelu_op = PReLU().set_device('CPU')

prelu_grad_op = PReLUGrad().set_device('CPU')

prod_ext_op = ProdExt().set_device('CPU')

quant_v2_op = QuantV2().set_device('CPU')

rand_ext_op = RandExt().set_device('CPU')

rand_like_ext_op = RandLikeExt().set_device('CPU')

randint_op = RandInt().set_device('CPU')

randint_like_op = RandIntLike().set_device('CPU')

randn_op = Randn().set_device('CPU')

randn_like_op = RandnLike().set_device('CPU')

randperm_ext_op = RandpermExt().set_device('CPU')

reciprocal_op = Reciprocal().set_device('CPU')

reflection_pad_1d_op = ReflectionPad1D().set_device('CPU')

reflection_pad_1d_grad_op = ReflectionPad1DGrad().set_device('CPU')

reflection_pad_2d_op = ReflectionPad2D().set_device('CPU')

reflection_pad_2d_grad_op = ReflectionPad2DGrad().set_device('CPU')

reflection_pad_3d_op = ReflectionPad3D().set_device('CPU')

reflection_pad_3d_grad_op = ReflectionPad3DGrad().set_device('CPU')

relu_op = ReLU().set_device('CPU')

relu_grad_op = ReluGrad().set_device('CPU')

remainder_scalar_tensor_op = RemainderScalarTensor().set_device('CPU')

remainder_tensor_scalar_op = RemainderTensorScalar().set_device('CPU')

remainder_tensor_tensor_op = RemainderTensorTensor().set_device('CPU')

repeat_op = Repeat().set_device('CPU')

repeat_interleave_grad_op = RepeatInterleaveGrad().set_device('CPU')

repeat_interleave_int_op = RepeatInterleaveInt().set_device('CPU')

repeat_interleave_tensor_op = RepeatInterleaveTensor().set_device('CPU')

replication_pad_1d_op = ReplicationPad1D().set_device('CPU')

replication_pad_1d_grad_op = ReplicationPad1DGrad().set_device('CPU')

replication_pad_2d_op = ReplicationPad2D().set_device('CPU')

replication_pad_2d_grad_op = ReplicationPad2DGrad().set_device('CPU')

replication_pad_3d_op = ReplicationPad3D().set_device('CPU')

replication_pad_3d_grad_op = ReplicationPad3DGrad().set_device('CPU')

reshape_op = Reshape().set_device('CPU')

rms_norm_grad_op = RmsNormGrad().set_device('CPU')

rotary_position_embedding_op = RotaryPositionEmbedding().set_device('CPU')

rotary_position_embedding_grad_op = RotaryPositionEmbeddingGrad().set_device('CPU')

round_op = Round().set_device('CPU')

rsqrt_op = Rsqrt().set_device('CPU')

scatter_op = Scatter().set_device('CPU')

scatter_add_ext_op = ScatterAddExt().set_device('CPU')

scatter_value_op = ScatterValue().set_device('CPU')

select_op = Select().set_device('CPU')

select_ext_view_op = SelectExtView().set_device('CPU')

select_v2_op = SelectV2().set_device('CPU')

selu_ext_op = SeLUExt().set_device('CPU')

selu_grad_op = SeluGrad().set_device('CPU')

sigmoid_op = Sigmoid().set_device('CPU')

sigmoid_grad_op = SigmoidGrad().set_device('CPU')

sign_op = Sign().set_device('CPU')

silent_check_v2_op = SilentCheckV2().set_device('CPU')

silent_check_v3_op = SilentCheckV3().set_device('CPU')

silu_op = SiLU().set_device('CPU')

silu_grad_op = SiLUGrad().set_device('CPU')

sin_op = Sin().set_device('CPU')

sinc_op = Sinc().set_device('CPU')

sinh_op = Sinh().set_device('CPU')

slice_op = Slice().set_device('CPU')

slice_ext_op = SliceExt().set_device('CPU')

slice_ext_view_op = SliceExtView().set_device('CPU')

softmax_backward_op = SoftmaxBackward().set_device('CPU')

softplus_ext_op = SoftplusExt().set_device('CPU')

softplus_grad_ext_op = SoftplusGradExt().set_device('CPU')

sort_ext_op = SortExt().set_device('CPU')

speed_fusion_attention_op = SpeedFusionAttention().set_device('CPU')

speed_fusion_attention_grad_op = SpeedFusionAttentionGrad().set_device('CPU')

split_tensor_op = SplitTensor().set_device('CPU')

split_tensor_view_op = SplitTensorView().set_device('CPU')

split_with_size_op = SplitWithSize().set_device('CPU')

split_with_size_view_op = SplitWithSizeView().set_device('CPU')

sqrt_op = Sqrt().set_device('CPU')

square_op = Square().set_device('CPU')

std_op = Std().set_device('CPU')

std_mean_op = StdMean().set_device('CPU')

sub_op = Sub().set_device('CPU')

sub_ext_op = SubExt().set_device('CPU')

sub_scalar_op = SubScalar().set_device('CPU')

sum_ext_op = SumExt().set_device('CPU')

swiglu_op = Swiglu().set_device('CPU')

swiglu_grad_op = SwigluGrad().set_device('CPU')

t_ext_op = TExt().set_device('CPU')

take_op = Take().set_device('CPU')

tan_op = Tan().set_device('CPU')

tanh_op = Tanh().set_device('CPU')

tanh_grad_op = TanhGrad().set_device('CPU')

threshold_op = Threshold().set_device('CPU')

threshold_grad_op = ThresholdGrad().set_device('CPU')

topk_ext_op = TopkExt().set_device('CPU')

trace_ext_op = TraceExt().set_device('CPU')

transpose_op = Transpose().set_device('CPU')

transpose_ext_view_op = TransposeExtView().set_device('CPU')

transpose_view_op = TransposeView().set_device('CPU')

triangular_solve_op = TriangularSolve().set_device('CPU')

tril_ext_op = TrilExt().set_device('CPU')

trunc_op = Trunc().set_device('CPU')

uniform_ext_op = UniformExt().set_device('CPU')

unique2_op = Unique2().set_device('CPU')

unique_dim_op = UniqueDim().set_device('CPU')

unstack_ext_view_op = UnstackExtView().set_device('CPU')

upsample_bicubic2d_op = UpsampleBicubic2D().set_device('CPU')

upsample_bicubic2d_grad_op = UpsampleBicubic2DGrad().set_device('CPU')

upsample_bilinear2d_op = UpsampleBilinear2D().set_device('CPU')

upsample_bilinear2d_grad_op = UpsampleBilinear2DGrad().set_device('CPU')

upsample_linear1d_op = UpsampleLinear1D().set_device('CPU')

upsample_linear1d_grad_op = UpsampleLinear1DGrad().set_device('CPU')

upsample_nearest1d_op = UpsampleNearest1D().set_device('CPU')

upsample_nearest1d_grad_op = UpsampleNearest1DGrad().set_device('CPU')

upsample_nearest2d_op = UpsampleNearest2D().set_device('CPU')

upsample_nearest2d_grad_op = UpsampleNearest2DGrad().set_device('CPU')

upsample_nearest3d_op = UpsampleNearest3D().set_device('CPU')

upsample_nearest3d_grad_op = UpsampleNearest3DGrad().set_device('CPU')

var_op = Var().set_device('CPU')

var_mean_op = VarMean().set_device('CPU')

view_as_op = ViewAs().set_device('CPU')

xlogy_op = Xlogy().set_device('CPU')

xlogy_scalar_other_op = XLogYScalarOther().set_device('CPU')

xlogy_scalar_self_op = XLogYScalarSelf().set_device('CPU')

zeros_like_ext_op = ZerosLikeExt().set_device('CPU')

