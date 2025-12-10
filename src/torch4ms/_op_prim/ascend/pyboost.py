from mindspore.ops.auto_generate.gen_ops_prim import *
from mindspore.ops.auto_generate.pyboost_inner_prim import *

abs_op = Abs().set_device('Ascend')

acos_ext_op = AcosExt().set_device('Ascend')

acosh_ext_op = AcoshExt().set_device('Ascend')

adamw_op = AdamW().set_device('Ascend')

adaptive_avg_pool1d_op = AdaptiveAvgPool1D().set_device('Ascend')

adaptive_avg_pool2d_ext_op = AdaptiveAvgPool2DExt().set_device('Ascend')

adaptive_avg_pool2d_grad_ext_op = AdaptiveAvgPool2DGradExt().set_device('Ascend')

adaptive_avg_pool3d_ext_op = AdaptiveAvgPool3DExt().set_device('Ascend')

adaptive_avg_pool3d_grad_ext_op = AdaptiveAvgPool3DGradExt().set_device('Ascend')

adaptive_max_pool1d_op = AdaptiveMaxPool1D().set_device('Ascend')

add_op = Add().set_device('Ascend')

add_ext_op = AddExt().set_device('Ascend')

add_layer_norm_grad_op = AddLayerNormGrad().set_device('Ascend')

add_layernorm_v2_op = AddLayerNormV2().set_device('Ascend')

add_rms_norm_op = AddRmsNorm().set_device('Ascend')

add_scalar_op = AddScalar().set_device('Ascend')

addbmm_op = Addbmm().set_device('Ascend')

addcdiv_ext_op = AddcdivExt().set_device('Ascend')

addcmul_ext_op = AddcmulExt().set_device('Ascend')

addmm_op = Addmm().set_device('Ascend')

addmv_op = Addmv().set_device('Ascend')

all_gather_matmul_op = AllGatherMatmul().set_device('Ascend')

arange_op = Arange().set_device('Ascend')

argmax_ext_op = ArgMaxExt().set_device('Ascend')

argmin_ext_op = ArgMinExt().set_device('Ascend')

argsort_op = ArgSort().set_device('Ascend')

as_strided_op = AsStrided().set_device('Ascend')

asin_ext_op = AsinExt().set_device('Ascend')

asinh_ext_op = AsinhExt().set_device('Ascend')

atan2_ext_op = Atan2Ext().set_device('Ascend')

atan_ext_op = AtanExt().set_device('Ascend')

atanh_op = Atanh().set_device('Ascend')

avg_pool1d_op = AvgPool1D().set_device('Ascend')

avg_pool2d_op = AvgPool2D().set_device('Ascend')

avg_pool2d_grad_op = AvgPool2DGrad().set_device('Ascend')

avg_pool3d_ext_op = AvgPool3DExt().set_device('Ascend')

avg_pool3d_grad_ext_op = AvgPool3DGradExt().set_device('Ascend')

baddbmm_op = Baddbmm().set_device('Ascend')

batch_norm_elemt_op = BatchNormElemt().set_device('Ascend')

batch_norm_elemt_grad_op = BatchNormElemtGrad().set_device('Ascend')

batch_norm_ext_op = BatchNormExt().set_device('Ascend')

batch_norm_gather_stats_with_counts_op = BatchNormGatherStatsWithCounts().set_device('Ascend')

batch_norm_reduce_grad_op = BatchNormReduceGrad().set_device('Ascend')

batch_norm_stats_op = BatchNormStats().set_device('Ascend')

bernoulli_ext_op = BernoulliExt().set_device('Ascend')

binary_cross_entropy_with_logits_backward_op = BinaryCrossEntropyWithLogitsBackward().set_device('Ascend')

bincount_ext_op = BincountExt().set_device('Ascend')

bitwise_and_scalar_op = BitwiseAndScalar().set_device('Ascend')

bitwise_and_tensor_op = BitwiseAndTensor().set_device('Ascend')

bitwise_not_op = BitwiseNot().set_device('Ascend')

bitwise_or_scalar_op = BitwiseOrScalar().set_device('Ascend')

bitwise_or_tensor_op = BitwiseOrTensor().set_device('Ascend')

bitwise_xor_scalar_op = BitwiseXorScalar().set_device('Ascend')

bitwise_xor_tensor_op = BitwiseXorTensor().set_device('Ascend')

bmm_ext_op = BatchMatMulExt().set_device('Ascend')

broadcast_to_view_op = BroadcastToView().set_device('Ascend')

ceil_op = Ceil().set_device('Ascend')

chunk_op = Chunk().set_device('Ascend')

chunk_view_op = ChunkView().set_device('Ascend')

clamp_scalar_op = ClampScalar().set_device('Ascend')

clamp_tensor_op = ClampTensor().set_device('Ascend')

clone_op = Clone().set_device('Ascend')

col2im_ext_op = Col2ImExt().set_device('Ascend')

col2im_grad_op = Col2ImGrad().set_device('Ascend')

constant_pad_nd_op = ConstantPadND().set_device('Ascend')

contiguous_op = Contiguous().set_device('Ascend')

conv1d_ext_op = Conv1DExt().set_device('Ascend')

conv1d_padding_op = Conv1DPadding().set_device('Ascend')

conv2d_ext_op = Conv2DExt().set_device('Ascend')

conv2d_padding_op = Conv2DPadding().set_device('Ascend')

conv3d_ext_op = Conv3DExt().set_device('Ascend')

conv3d_padding_op = Conv3DPadding().set_device('Ascend')

conv_transpose2d_op = ConvTranspose2D().set_device('Ascend')

convolution_op = Convolution().set_device('Ascend')

convolution_grad_op = ConvolutionGrad().set_device('Ascend')

convolution_str_op = ConvolutionStr().set_device('Ascend')

convolution_str_grad_op = ConvolutionStrGrad().set_device('Ascend')

copy_op = Copy().set_device('Ascend')

cos_op = Cos().set_device('Ascend')

cosh_op = Cosh().set_device('Ascend')

count_nonzero_op = CountNonZero().set_device('Ascend')

cummin_ext_op = CumminExt().set_device('Ascend')

cumsum_ext_op = CumsumExt().set_device('Ascend')

dense_op = Dense().set_device('Ascend')

diag_ext_op = DiagExt().set_device('Ascend')

dist_comm_all_gather_op = DistCommAllGather().set_device('Ascend')

dist_comm_all_gather_into_tensor_op = DistCommAllGatherIntoTensor().set_device('Ascend')

dist_comm_all_reduce_op = DistCommAllReduce().set_device('Ascend')

dist_comm_all_to_all_v_op = DistCommAllToAllV().set_device('Ascend')

dist_comm_all_to_all_v_single_op = DistCommAllToAllVSingle().set_device('Ascend')

dist_comm_barrier_op = DistCommBarrier().set_device('Ascend')

dist_comm_batch_isend_irecv_op = DistCommBatchIsendIrecv().set_device('Ascend')

dist_comm_broadcast_op = DistCommBroadcast().set_device('Ascend')

dist_comm_gather_op = DistCommGather().set_device('Ascend')

dist_comm_gather_into_tensor_op = DistCommGatherIntoTensor().set_device('Ascend')

dist_comm_irecv_op = DistCommIrecv().set_device('Ascend')

dist_comm_isend_op = DistCommIsend().set_device('Ascend')

dist_comm_reduce_op = DistCommReduce().set_device('Ascend')

dist_comm_reduce_scatter_op = DistCommReduceScatter().set_device('Ascend')

dist_comm_reduce_scatter_tensor_op = DistCommReduceScatterTensor().set_device('Ascend')

dist_comm_scatter_op = DistCommScatter().set_device('Ascend')

dist_comm_scatter_tensor_op = DistCommScatterTensor().set_device('Ascend')

div_op = Div().set_device('Ascend')

divmod_op = DivMod().set_device('Ascend')

divmods_op = DivMods().set_device('Ascend')

divs_op = Divs().set_device('Ascend')

dot_op = Dot().set_device('Ascend')

dropout_do_mask_ext_op = DropoutDoMaskExt().set_device('Ascend')

dropout_ext_op = DropoutExt().set_device('Ascend')

dropout_gen_mask_ext_op = DropoutGenMaskExt().set_device('Ascend')

dropout_grad_ext_op = DropoutGradExt().set_device('Ascend')

dynamic_quant_ext_op = DynamicQuantExt().set_device('Ascend')

elu_grad_ext_op = EluGradExt().set_device('Ascend')

embedding_op = Embedding().set_device('Ascend')

embedding_dense_backward_op = EmbeddingDenseBackward().set_device('Ascend')

equal_op = Equal().set_device('Ascend')

equal_ext_op = EqualExt().set_device('Ascend')

erf_op = Erf().set_device('Ascend')

erfc_op = Erfc().set_device('Ascend')

erfinv_op = Erfinv().set_device('Ascend')

exp_op = Exp().set_device('Ascend')

exp2_op = Exp2().set_device('Ascend')

expand_as_op = ExpandAs().set_device('Ascend')

expand_dims_op = ExpandDims().set_device('Ascend')

expand_dims_view_op = ExpandDimsView().set_device('Ascend')

expm1_op = Expm1().set_device('Ascend')

eye_op = Eye().set_device('Ascend')

fill_scalar_op = FillScalar().set_device('Ascend')

fill_tensor_op = FillTensor().set_device('Ascend')

flatten_ext_op = FlattenExt().set_device('Ascend')

floor_op = Floor().set_device('Ascend')

floor_div_op = FloorDiv().set_device('Ascend')

floor_div_scalar_op = FloorDivScalar().set_device('Ascend')

fmod_scalar_op = FmodScalar().set_device('Ascend')

fmod_tensor_op = FmodTensor().set_device('Ascend')

frac_op = Frac().set_device('Ascend')

full_like_op = FullLike().set_device('Ascend')

gather_d_op = GatherD().set_device('Ascend')

gather_d_grad_v2_op = GatherDGradV2().set_device('Ascend')

gcd_op = Gcd().set_device('Ascend')

gelu_op = GeLU().set_device('Ascend')

gelu_ext_op = GeluExt().set_device('Ascend')

gelu_grad_op = GeLUGrad().set_device('Ascend')

gelu_grad_ext_op = GeluGradExt().set_device('Ascend')

generator_op = Generator().set_device('Ascend')

gmm_op = Gmm().set_device('Ascend')

gmm_backward_op = GmmBackward().set_device('Ascend')

gmm_backward_fusion_op = GmmBackwardFusion().set_device('Ascend')

gmm_v2_op = GmmV2().set_device('Ascend')

gmm_v2_backward_op = GmmV2Backward().set_device('Ascend')

gmm_v2_backward_fusion_op = GmmV2BackwardFusion().set_device('Ascend')

greater_op = Greater().set_device('Ascend')

greater_equal_op = GreaterEqual().set_device('Ascend')

greater_equal_scalar_op = GreaterEqualScalar().set_device('Ascend')

group_norm_op = GroupNorm().set_device('Ascend')

group_norm_grad_op = GroupNormGrad().set_device('Ascend')

grouped_matmul_v2_op = GroupedMatmulV2().set_device('Ascend')

grouped_matmul_v4_op = GroupedMatmulV4().set_device('Ascend')

hardtanh_op = Hardtanh().set_device('Ascend')

hardtanh_grad_op = HardtanhGrad().set_device('Ascend')

histc_ext_op = HistcExt().set_device('Ascend')

hsigmoid_op = HSigmoid().set_device('Ascend')

hsigmoid_grad_op = HSigmoidGrad().set_device('Ascend')

hswish_op = HSwish().set_device('Ascend')

hswish_grad_op = HSwishGrad().set_device('Ascend')

im2col_ext_op = Im2ColExt().set_device('Ascend')

index_op = Index().set_device('Ascend')

index_add_ext_op = IndexAddExt().set_device('Ascend')

index_fill_scalar_op = IndexFillScalar().set_device('Ascend')

index_fill_tensor_op = IndexFillTensor().set_device('Ascend')

index_select_op = IndexSelect().set_device('Ascend')

inner_comm_all_gather_op = InnerCommAllGather().set_device('Ascend')

inner_comm_all_reduce_op = InnerCommAllReduce().set_device('Ascend')

inner_comm_all_to_all_v_op = InnerCommAllToAllV().set_device('Ascend')

inner_comm_irecv_op = InnerCommIrecv().set_device('Ascend')

inner_comm_isend_op = InnerCommIsend().set_device('Ascend')

inner_comm_reduce_scatter_op = InnerCommReduceScatter().set_device('Ascend')

inner_index_op = InnerIndex().set_device('Ascend')

inner_inplace_index_put_op = InnerInplaceIndexPut().set_device('Ascend')

inner_non_zero_op = InnerNonZero().set_device('Ascend')

inplace_add_ext_op = InplaceAddExt().set_device('Ascend')

inplace_addmm_op = InplaceAddmm().set_device('Ascend')

inplace_adds_ext_op = InplaceAddsExt().set_device('Ascend')

inplace_clamp_scalar_op = InplaceClampScalar().set_device('Ascend')

inplace_clamp_tensor_op = InplaceClampTensor().set_device('Ascend')

inplace_copy_op = InplaceCopy().set_device('Ascend')

inplace_div_op = InplaceDiv().set_device('Ascend')

inplace_divmod_op = InplaceDivMod().set_device('Ascend')

inplace_divmods_op = InplaceDivMods().set_device('Ascend')

inplace_divs_op = InplaceDivs().set_device('Ascend')

inplace_elu_op = InplaceElu().set_device('Ascend')

inplace_erfinv_op = InplaceErfinv().set_device('Ascend')

inplace_exp_op = InplaceExp().set_device('Ascend')

inplace_exponential_op = InplaceExponential().set_device('Ascend')

inplace_fill_diagonal_op = InplaceFillDiagonal().set_device('Ascend')

inplace_fill_scalar_op = InplaceFillScalar().set_device('Ascend')

inplace_fill_tensor_op = InplaceFillTensor().set_device('Ascend')

inplace_floor_op = InplaceFloor().set_device('Ascend')

inplace_floor_divide_op = InplaceFloorDivide().set_device('Ascend')

inplace_floor_divides_op = InplaceFloorDivides().set_device('Ascend')

inplace_grouped_matmul_add_op = InplaceGroupedMatmulAdd().set_device('Ascend')

inplace_hardtanh_op = InplaceHardtanh().set_device('Ascend')

inplace_index_add_op = InplaceIndexAddExt().set_device('Ascend')

inplace_index_put_op = InplaceIndexPut().set_device('Ascend')

inplace_log_op = InplaceLog().set_device('Ascend')

inplace_masked_fill_scalar_op = InplaceMaskedFillScalar().set_device('Ascend')

inplace_masked_fill_tensor_op = InplaceMaskedFillTensor().set_device('Ascend')

inplace_mul_op = InplaceMul().set_device('Ascend')

inplace_muls_op = InplaceMuls().set_device('Ascend')

inplace_normal_op = InplaceNormal().set_device('Ascend')

inplace_put_op = InplacePut().set_device('Ascend')

inplace_random_op = InplaceRandom().set_device('Ascend')

inplace_relu_op = InplaceReLU().set_device('Ascend')

inplace_scatter_add_op = InplaceScatterAdd().set_device('Ascend')

inplace_scatter_src_op = InplaceScatterSrc().set_device('Ascend')

inplace_scatter_src_reduce_op = InplaceScatterSrcReduce().set_device('Ascend')

inplace_scatter_value_op = InplaceScatterValue().set_device('Ascend')

inplace_scatter_value_reduce_op = InplaceScatterValueReduce().set_device('Ascend')

inplace_stop_gradient_op = InplaceStopGradient().set_device('Ascend')

inplace_sub_ext_op = InplaceSubExt().set_device('Ascend')

inplace_sub_scalar_op = InplaceSubScalar().set_device('Ascend')

inplace_tanh_op = InplaceTanh().set_device('Ascend')

inplace_threshold_op = InplaceThreshold().set_device('Ascend')

inplace_uniform_op = InplaceUniform().set_device('Ascend')

inplace_zero_op = InplaceZero().set_device('Ascend')

isfinite_op = IsFinite().set_device('Ascend')

isinf_op = IsInf().set_device('Ascend')

isneginf_op = IsNegInf().set_device('Ascend')

kl_div_op = KLDiv().set_device('Ascend')

kl_div_grad_op = KLDivGrad().set_device('Ascend')

kthvalue_op = Kthvalue().set_device('Ascend')

kv_cache_scatter_update_op = KVCacheScatterUpdate().set_device('Ascend')

l1_loss_backward_ext_op = L1LossBackwardExt().set_device('Ascend')

l1_loss_ext_op = L1LossExt().set_device('Ascend')

layer_norm_ext_op = LayerNormExt().set_device('Ascend')

layer_norm_grad_ext_op = LayerNormGradExt().set_device('Ascend')

leaky_relu_ext_op = LeakyReLUExt().set_device('Ascend')

leaky_relu_grad_ext_op = LeakyReLUGradExt().set_device('Ascend')

lerp_op = Lerp().set_device('Ascend')

lerp_scalar_op = LerpScalar().set_device('Ascend')

less_op = Less().set_device('Ascend')

less_equal_op = LessEqual().set_device('Ascend')

lin_space_ext_op = LinSpaceExt().set_device('Ascend')

linalg_qr_op = LinalgQr().set_device('Ascend')

linalg_vector_norm_op = LinalgVectorNorm().set_device('Ascend')

log_op = Log().set_device('Ascend')

log10_op = Log10().set_device('Ascend')

log1p_op = Log1p().set_device('Ascend')

log2_op = Log2().set_device('Ascend')

log_softmax_ext_op = LogSoftmaxExt().set_device('Ascend')

logaddexp_op = LogAddExp().set_device('Ascend')

logaddexp2_op = LogAddExp2().set_device('Ascend')

logical_and_op = LogicalAnd().set_device('Ascend')

logical_not_op = LogicalNot().set_device('Ascend')

logical_or_op = LogicalOr().set_device('Ascend')

logical_xor_op = LogicalXor().set_device('Ascend')

logsigmoid_op = LogSigmoid().set_device('Ascend')

logsigmoid_grad_op = LogSigmoidGrad().set_device('Ascend')

logsumexp_op = LogSumExp().set_device('Ascend')

masked_fill_op = MaskedFill().set_device('Ascend')

masked_select_op = MaskedSelect().set_device('Ascend')

masked_select_grad_op = MaskedSelectGrad().set_device('Ascend')

matmul_allreduce_add_rmsnorm_op = MatmulAllReduceAddRmsNorm().set_device('Ascend')

matmul_ext_op = MatMulExt().set_device('Ascend')

matmul_reduce_scatter_op = MatmulReduceScatter().set_device('Ascend')

matrix_inverse_ext_op = MatrixInverseExt().set_device('Ascend')

max_op = Max().set_device('Ascend')

max_dim_op = MaxDim().set_device('Ascend')

max_unpool2d_ext_op = MaxUnpool2DExt().set_device('Ascend')

maximum_op = Maximum().set_device('Ascend')

mean_ext_op = MeanExt().set_device('Ascend')

median_dim_op = MedianDim().set_device('Ascend')

median_ext_op = MedianExt().set_device('Ascend')

min_op = Min().set_device('Ascend')

min_dim_op = MinDim().set_device('Ascend')

minimum_op = Minimum().set_device('Ascend')

mish_ext_op = MishExt().set_device('Ascend')

mish_grad_ext_op = MishGradExt().set_device('Ascend')

mm_ext_op = Mm().set_device('Ascend')

moe_compute_expert_tokens_op = MoeComputeExpertTokens().set_device('Ascend')

moe_finalize_routing_op = MoeFinalizeRouting().set_device('Ascend')

moe_gating_top_k_softmax_op = MoeGatingTopKSoftmax().set_device('Ascend')

moe_init_routing_op = MoeInitRouting().set_device('Ascend')

moe_init_routing_v2_op = MoeInitRoutingV2().set_device('Ascend')

moe_token_permute_op = MoeTokenPermute().set_device('Ascend')

moe_token_permute_grad_op = MoeTokenPermuteGrad().set_device('Ascend')

moe_token_unpermute_op = MoeTokenUnpermute().set_device('Ascend')

moe_token_unpermute_grad_op = MoeTokenUnpermuteGrad().set_device('Ascend')

mse_loss_ext_op = MSELossExt().set_device('Ascend')

mse_loss_grad_ext_op = MSELossGradExt().set_device('Ascend')

mul_op = Mul().set_device('Ascend')

muls_op = Muls().set_device('Ascend')

multi_scale_deformable_attn_op = MultiScaleDeformableAttn().set_device('Ascend')

multi_scale_deformable_attn_grad_op = MultiScaleDeformableAttnGrad().set_device('Ascend')

multinomial_ext_op = MultinomialExt().set_device('Ascend')

mv_op = Mv().set_device('Ascend')

nansum_op = Nansum().set_device('Ascend')

narrow_op = Narrow().set_device('Ascend')

narrow_view_op = NarrowView().set_device('Ascend')

neg_op = Neg().set_device('Ascend')

new_ones_op = NewOnes().set_device('Ascend')

new_zeros_op = NewZeros().set_device('Ascend')

nllloss_2d_op = NLLLoss2d().set_device('Ascend')

nllloss_2d_grad_op = NLLLoss2dGrad().set_device('Ascend')

non_zero_op = NonZero().set_device('Ascend')

non_zero_ext_op = NonZeroExt().set_device('Ascend')

norm_op = Norm().set_device('Ascend')

normal_float_float_op = NormalFloatFloat().set_device('Ascend')

normal_float_tensor_op = NormalFloatTensor().set_device('Ascend')

normal_tensor_float_op = NormalTensorFloat().set_device('Ascend')

normal_tensor_tensor_op = NormalTensorTensor().set_device('Ascend')

not_equal_op = NotEqual().set_device('Ascend')

ones_like_ext_op = OnesLikeExt().set_device('Ascend')

outer_op = Outer().set_device('Ascend')

pixel_shuffle_op = PixelShuffle().set_device('Ascend')

polar_op = Polar().set_device('Ascend')

pow_op = Pow().set_device('Ascend')

pow_scalar_tensor_op = PowScalarTensor().set_device('Ascend')

pow_tensor_scalar_op = PowTensorScalar().set_device('Ascend')

prelu_op = PReLU().set_device('Ascend')

prelu_grad_op = PReLUGrad().set_device('Ascend')

prod_ext_op = ProdExt().set_device('Ascend')

quant_v2_op = QuantV2().set_device('Ascend')

rand_ext_op = RandExt().set_device('Ascend')

rand_like_ext_op = RandLikeExt().set_device('Ascend')

randint_op = RandInt().set_device('Ascend')

randint_like_op = RandIntLike().set_device('Ascend')

randn_op = Randn().set_device('Ascend')

randn_like_op = RandnLike().set_device('Ascend')

randperm_ext_op = RandpermExt().set_device('Ascend')

reciprocal_op = Reciprocal().set_device('Ascend')

reflection_pad_1d_op = ReflectionPad1D().set_device('Ascend')

reflection_pad_1d_grad_op = ReflectionPad1DGrad().set_device('Ascend')

reflection_pad_2d_op = ReflectionPad2D().set_device('Ascend')

reflection_pad_2d_grad_op = ReflectionPad2DGrad().set_device('Ascend')

reflection_pad_3d_op = ReflectionPad3D().set_device('Ascend')

reflection_pad_3d_grad_op = ReflectionPad3DGrad().set_device('Ascend')

relu_op = ReLU().set_device('Ascend')

relu_grad_op = ReluGrad().set_device('Ascend')

remainder_scalar_tensor_op = RemainderScalarTensor().set_device('Ascend')

remainder_tensor_scalar_op = RemainderTensorScalar().set_device('Ascend')

remainder_tensor_tensor_op = RemainderTensorTensor().set_device('Ascend')

repeat_op = Repeat().set_device('Ascend')

repeat_interleave_grad_op = RepeatInterleaveGrad().set_device('Ascend')

repeat_interleave_int_op = RepeatInterleaveInt().set_device('Ascend')

repeat_interleave_tensor_op = RepeatInterleaveTensor().set_device('Ascend')

replication_pad_1d_op = ReplicationPad1D().set_device('Ascend')

replication_pad_1d_grad_op = ReplicationPad1DGrad().set_device('Ascend')

replication_pad_2d_op = ReplicationPad2D().set_device('Ascend')

replication_pad_2d_grad_op = ReplicationPad2DGrad().set_device('Ascend')

replication_pad_3d_op = ReplicationPad3D().set_device('Ascend')

replication_pad_3d_grad_op = ReplicationPad3DGrad().set_device('Ascend')

reshape_op = Reshape().set_device('Ascend')

rms_norm_grad_op = RmsNormGrad().set_device('Ascend')

rotary_position_embedding_op = RotaryPositionEmbedding().set_device('Ascend')

rotary_position_embedding_grad_op = RotaryPositionEmbeddingGrad().set_device('Ascend')

round_op = Round().set_device('Ascend')

rsqrt_op = Rsqrt().set_device('Ascend')

scatter_op = Scatter().set_device('Ascend')

scatter_add_ext_op = ScatterAddExt().set_device('Ascend')

scatter_value_op = ScatterValue().set_device('Ascend')

select_op = Select().set_device('Ascend')

select_ext_view_op = SelectExtView().set_device('Ascend')

select_v2_op = SelectV2().set_device('Ascend')

selu_ext_op = SeLUExt().set_device('Ascend')

selu_grad_op = SeluGrad().set_device('Ascend')

sigmoid_op = Sigmoid().set_device('Ascend')

sigmoid_grad_op = SigmoidGrad().set_device('Ascend')

sign_op = Sign().set_device('Ascend')

silent_check_v2_op = SilentCheckV2().set_device('Ascend')

silent_check_v3_op = SilentCheckV3().set_device('Ascend')

silu_op = SiLU().set_device('Ascend')

silu_grad_op = SiLUGrad().set_device('Ascend')

sin_op = Sin().set_device('Ascend')

sinc_op = Sinc().set_device('Ascend')

sinh_op = Sinh().set_device('Ascend')

slice_op = Slice().set_device('Ascend')

slice_ext_op = SliceExt().set_device('Ascend')

slice_ext_view_op = SliceExtView().set_device('Ascend')

softmax_backward_op = SoftmaxBackward().set_device('Ascend')

softplus_ext_op = SoftplusExt().set_device('Ascend')

softplus_grad_ext_op = SoftplusGradExt().set_device('Ascend')

sort_ext_op = SortExt().set_device('Ascend')

speed_fusion_attention_op = SpeedFusionAttention().set_device('Ascend')

speed_fusion_attention_grad_op = SpeedFusionAttentionGrad().set_device('Ascend')

split_tensor_op = SplitTensor().set_device('Ascend')

split_tensor_view_op = SplitTensorView().set_device('Ascend')

split_with_size_op = SplitWithSize().set_device('Ascend')

split_with_size_view_op = SplitWithSizeView().set_device('Ascend')

sqrt_op = Sqrt().set_device('Ascend')

square_op = Square().set_device('Ascend')

std_op = Std().set_device('Ascend')

std_mean_op = StdMean().set_device('Ascend')

sub_op = Sub().set_device('Ascend')

sub_ext_op = SubExt().set_device('Ascend')

sub_scalar_op = SubScalar().set_device('Ascend')

sum_ext_op = SumExt().set_device('Ascend')

swiglu_op = Swiglu().set_device('Ascend')

swiglu_grad_op = SwigluGrad().set_device('Ascend')

t_ext_op = TExt().set_device('Ascend')

take_op = Take().set_device('Ascend')

tan_op = Tan().set_device('Ascend')

tanh_op = Tanh().set_device('Ascend')

tanh_grad_op = TanhGrad().set_device('Ascend')

threshold_op = Threshold().set_device('Ascend')

threshold_grad_op = ThresholdGrad().set_device('Ascend')

topk_ext_op = TopkExt().set_device('Ascend')

trace_ext_op = TraceExt().set_device('Ascend')

transpose_op = Transpose().set_device('Ascend')

transpose_ext_view_op = TransposeExtView().set_device('Ascend')

transpose_view_op = TransposeView().set_device('Ascend')

triangular_solve_op = TriangularSolve().set_device('Ascend')

tril_ext_op = TrilExt().set_device('Ascend')

trunc_op = Trunc().set_device('Ascend')

uniform_ext_op = UniformExt().set_device('Ascend')

unique2_op = Unique2().set_device('Ascend')

unique_dim_op = UniqueDim().set_device('Ascend')

unstack_ext_view_op = UnstackExtView().set_device('Ascend')

upsample_bicubic2d_op = UpsampleBicubic2D().set_device('Ascend')

upsample_bicubic2d_grad_op = UpsampleBicubic2DGrad().set_device('Ascend')

upsample_bilinear2d_op = UpsampleBilinear2D().set_device('Ascend')

upsample_bilinear2d_grad_op = UpsampleBilinear2DGrad().set_device('Ascend')

upsample_linear1d_op = UpsampleLinear1D().set_device('Ascend')

upsample_linear1d_grad_op = UpsampleLinear1DGrad().set_device('Ascend')

upsample_nearest1d_op = UpsampleNearest1D().set_device('Ascend')

upsample_nearest1d_grad_op = UpsampleNearest1DGrad().set_device('Ascend')

upsample_nearest2d_op = UpsampleNearest2D().set_device('Ascend')

upsample_nearest2d_grad_op = UpsampleNearest2DGrad().set_device('Ascend')

upsample_nearest3d_op = UpsampleNearest3D().set_device('Ascend')

upsample_nearest3d_grad_op = UpsampleNearest3DGrad().set_device('Ascend')

var_op = Var().set_device('Ascend')

var_mean_op = VarMean().set_device('Ascend')

view_as_op = ViewAs().set_device('Ascend')

xlogy_op = Xlogy().set_device('Ascend')

xlogy_scalar_other_op = XLogYScalarOther().set_device('Ascend')

xlogy_scalar_self_op = XLogYScalarSelf().set_device('Ascend')

zeros_like_ext_op = ZerosLikeExt().set_device('Ascend')

