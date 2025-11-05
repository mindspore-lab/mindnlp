from mindspore.ops.operations import *
from mindspore.ops.operations._grad_ops import *
from mindspore.ops._primitive_cache import _get_cache_prim


a_cos_grad_op = ACosGrad()
def a_cos_grad(*args):
    a_cos_grad_op = hook_call(a_cos_grad_op)
    return a_cos_grad_op(*args)


abs_grad_op = AbsGrad()
def abs_grad(*args):
    abs_grad_op = hook_call(abs_grad_op)
    return abs_grad_op(*args)


acosh_grad_op = AcoshGrad()
def acosh_grad(*args):
    acosh_grad_op = hook_call(acosh_grad_op)
    return acosh_grad_op(*args)


adaptive_avg_pool2_d_grad_op = AdaptiveAvgPool2DGrad()
def adaptive_avg_pool2_d_grad(*args):
    adaptive_avg_pool2_d_grad_op = hook_call(adaptive_avg_pool2_d_grad_op)
    return adaptive_avg_pool2_d_grad_op(*args)


adaptive_avg_pool3_d_grad_op = AdaptiveAvgPool3DGrad()
def adaptive_avg_pool3_d_grad(*args):
    adaptive_avg_pool3_d_grad_op = hook_call(adaptive_avg_pool3_d_grad_op)
    return adaptive_avg_pool3_d_grad_op(*args)


adaptive_max_pool2_d_grad_op = AdaptiveMaxPool2DGrad()
def adaptive_max_pool2_d_grad(*args):
    adaptive_max_pool2_d_grad_op = hook_call(adaptive_max_pool2_d_grad_op)
    return adaptive_max_pool2_d_grad_op(*args)


adaptive_max_pool3_d_grad_op = AdaptiveMaxPool3DGrad()
def adaptive_max_pool3_d_grad(*args):
    adaptive_max_pool3_d_grad_op = hook_call(adaptive_max_pool3_d_grad_op)
    return adaptive_max_pool3_d_grad_op(*args)


def affine_grid_grad(*args):
    op = _get_cache_prim(AffineGridGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


asin_grad_op = AsinGrad()
def asin_grad(*args):
    asin_grad_op = hook_call(asin_grad_op)
    return asin_grad_op(*args)


asinh_grad_op = AsinhGrad()
def asinh_grad(*args):
    asinh_grad_op = hook_call(asinh_grad_op)
    return asinh_grad_op(*args)


atan_grad_op = AtanGrad()
def atan_grad(*args):
    atan_grad_op = hook_call(atan_grad_op)
    return atan_grad_op(*args)


def avg_pool3_d_grad(*args):
    op = _get_cache_prim(AvgPool3DGrad)(*args[-8:])
    op = hook_call(op)
    return op(*args[:-8])


def avg_pool_grad(*args):
    op = _get_cache_prim(AvgPoolGrad)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def avg_pool_grad_ge(*args):
    op = _get_cache_prim(AvgPoolGradGe)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def avg_pool_grad_v1(*args):
    op = _get_cache_prim(AvgPoolGradV1)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def avg_pool_grad_vm(*args):
    op = _get_cache_prim(AvgPoolGradVm)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def bn_training_reduce_grad(*args):
    op = _get_cache_prim(BNTrainingReduceGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def bn_training_update_grad(*args):
    op = _get_cache_prim(BNTrainingUpdateGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def basic_lstm_cell_c_state_grad(*args):
    op = _get_cache_prim(BasicLSTMCellCStateGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def basic_lstm_cell_input_grad(*args):
    op = _get_cache_prim(BasicLSTMCellInputGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


basic_lstm_cell_weight_grad_op = BasicLSTMCellWeightGrad()
def basic_lstm_cell_weight_grad(*args):
    basic_lstm_cell_weight_grad_op = hook_call(basic_lstm_cell_weight_grad_op)
    return basic_lstm_cell_weight_grad_op(*args)


def batch_norm_grad(*args):
    op = _get_cache_prim(BatchNormGrad)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def batch_norm_grad_grad(*args):
    op = _get_cache_prim(BatchNormGradGrad)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def bias_add_grad(*args):
    op = _get_cache_prim(BiasAddGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def binary_cross_entropy_grad(*args):
    op = _get_cache_prim(BinaryCrossEntropyGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


cholesky_grad_op = CholeskyGrad()
def cholesky_grad(*args):
    cholesky_grad_op = hook_call(cholesky_grad_op)
    return cholesky_grad_op(*args)


def concat_offset(*args):
    op = _get_cache_prim(ConcatOffset)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def conv2_d_backprop_filter(*args):
    op = _get_cache_prim(Conv2DBackpropFilter)(*args[-10:])
    op = hook_call(op)
    return op(*args[:-10])


def conv3_d_backprop_filter(*args):
    op = _get_cache_prim(Conv3DBackpropFilter)(*args[-9:])
    op = hook_call(op)
    return op(*args[:-9])


def deformable_offsets_grad(*args):
    op = _get_cache_prim(DeformableOffsetsGrad)(*args[-7:])
    op = hook_call(op)
    return op(*args[:-7])


def depthwise_conv2d_native_backprop_filter(*args):
    op = _get_cache_prim(DepthwiseConv2dNativeBackpropFilter)(*args[-9:])
    op = hook_call(op)
    return op(*args[:-9])


def depthwise_conv2d_native_backprop_input(*args):
    op = _get_cache_prim(DepthwiseConv2dNativeBackpropInput)(*args[-9:])
    op = hook_call(op)
    return op(*args[:-9])


def dilation2_d_backprop_filter(*args):
    op = _get_cache_prim(Dilation2DBackpropFilter)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def dilation2_d_backprop_input(*args):
    op = _get_cache_prim(Dilation2DBackpropInput)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def dropout_grad(*args):
    op = _get_cache_prim(DropoutGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def dynamic_gruv2_grad(*args):
    op = _get_cache_prim(DynamicGRUV2Grad)(*args[-8:])
    op = hook_call(op)
    return op(*args[:-8])


def dynamic_rnn_grad(*args):
    op = _get_cache_prim(DynamicRNNGrad)(*args[-9:])
    op = hook_call(op)
    return op(*args[:-9])


def einsum_grad(*args):
    op = _get_cache_prim(EinsumGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


elu_grad_op = EluGrad()
def elu_grad(*args):
    elu_grad_op = hook_call(elu_grad_op)
    return elu_grad_op(*args)


embedding_lookup_comm_grad_op = EmbeddingLookupCommGrad()
def embedding_lookup_comm_grad(*args):
    embedding_lookup_comm_grad_op = hook_call(embedding_lookup_comm_grad_op)
    return embedding_lookup_comm_grad_op(*args)


fast_ge_lu_grad_op = FastGeLUGrad()
def fast_ge_lu_grad(*args):
    fast_ge_lu_grad_op = hook_call(fast_ge_lu_grad_op)
    return fast_ge_lu_grad_op(*args)


def flash_attention_score_grad(*args):
    op = _get_cache_prim(FlashAttentionScoreGrad)(*args[-8:])
    op = hook_call(op)
    return op(*args[:-8])


flatten_grad_op = FlattenGrad()
def flatten_grad(*args):
    flatten_grad_op = hook_call(flatten_grad_op)
    return flatten_grad_op(*args)


def fractional_avg_pool_grad(*args):
    op = _get_cache_prim(FractionalAvgPoolGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def fractional_max_pool3_d_grad_with_fixed_ksize(*args):
    op = _get_cache_prim(FractionalMaxPool3DGradWithFixedKsize)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def fractional_max_pool_grad(*args):
    op = _get_cache_prim(FractionalMaxPoolGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def fractional_max_pool_grad_with_fixed_ksize(*args):
    op = _get_cache_prim(FractionalMaxPoolGradWithFixedKsize)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def gruv2_grad(*args):
    op = _get_cache_prim(GRUV2Grad)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


gather_d_grad_v2_op = GatherDGradV2()
def gather_d_grad_v2(*args):
    gather_d_grad_v2_op = hook_call(gather_d_grad_v2_op)
    return gather_d_grad_v2_op(*args)


ge_lu_grad_op = GeLUGrad()
def ge_lu_grad(*args):
    ge_lu_grad_op = hook_call(ge_lu_grad_op)
    return ge_lu_grad_op(*args)


def global_comm(*args):
    op = _get_cache_prim(GlobalComm)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def glu_grad(*args):
    op = _get_cache_prim(GluGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def grid_sampler2_d_grad(*args):
    op = _get_cache_prim(GridSampler2DGrad)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def grid_sampler3_d_grad(*args):
    op = _get_cache_prim(GridSampler3DGrad)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def gru_grad_data(*args):
    op = _get_cache_prim(GruGradData)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def gru_grad_weight(*args):
    op = _get_cache_prim(GruGradWeight)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def h_shrink_grad(*args):
    op = _get_cache_prim(HShrinkGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


h_sigmoid_grad_op = HSigmoidGrad()
def h_sigmoid_grad(*args):
    h_sigmoid_grad_op = hook_call(h_sigmoid_grad_op)
    return h_sigmoid_grad_op(*args)


h_swish_grad_op = HSwishGrad()
def h_swish_grad(*args):
    h_swish_grad_op = hook_call(h_swish_grad_op)
    return h_swish_grad_op(*args)


igamma_grad_a_op = IgammaGradA()
def igamma_grad_a(*args):
    igamma_grad_a_op = hook_call(igamma_grad_a_op)
    return igamma_grad_a_op(*args)


def instance_norm_grad(*args):
    op = _get_cache_prim(InstanceNormGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def instance_norm_v2_grad(*args):
    op = _get_cache_prim(InstanceNormV2Grad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


inv_grad_op = InvGrad()
def inv_grad(*args):
    inv_grad_op = hook_call(inv_grad_op)
    return inv_grad_op(*args)


def kl_div_loss_grad(*args):
    op = _get_cache_prim(KLDivLossGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def l2_normalize_grad(*args):
    op = _get_cache_prim(L2NormalizeGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def lrn_grad(*args):
    op = _get_cache_prim(LRNGrad)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def lstm_grad(*args):
    op = _get_cache_prim(LSTMGrad)(*args[-7:])
    op = hook_call(op)
    return op(*args[:-7])


def lstm_grad_data(*args):
    op = _get_cache_prim(LSTMGradData)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def lstm_grad_weight(*args):
    op = _get_cache_prim(LSTMGradWeight)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def layer_norm_grad(*args):
    op = _get_cache_prim(LayerNormGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def layer_norm_grad_grad(*args):
    op = _get_cache_prim(LayerNormGradGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def log_softmax_grad(*args):
    op = _get_cache_prim(LogSoftmaxGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def logit_grad(*args):
    op = _get_cache_prim(LogitGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def lu_unpack_grad(*args):
    op = _get_cache_prim(LuUnpackGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


map_tensor_get_grad_op = MapTensorGetGrad()
def map_tensor_get_grad(*args):
    map_tensor_get_grad_op = hook_call(map_tensor_get_grad_op)
    return map_tensor_get_grad_op(*args)


masked_select_grad_op = MaskedSelectGrad()
def masked_select_grad(*args):
    masked_select_grad_op = hook_call(masked_select_grad_op)
    return masked_select_grad_op(*args)


def max_pool3_d_grad(*args):
    op = _get_cache_prim(MaxPool3DGrad)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


def max_pool3_d_grad_grad(*args):
    op = _get_cache_prim(MaxPool3DGradGrad)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def max_pool3_d_grad_with_argmax(*args):
    op = _get_cache_prim(MaxPool3DGradWithArgmax)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def max_pool_grad(*args):
    op = _get_cache_prim(MaxPoolGrad)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def max_pool_grad_grad(*args):
    op = _get_cache_prim(MaxPoolGradGrad)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def max_pool_grad_grad_with_argmax(*args):
    op = _get_cache_prim(MaxPoolGradGradWithArgmax)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def max_pool_grad_v1(*args):
    op = _get_cache_prim(MaxPoolGradV1)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def max_pool_grad_with_argmax(*args):
    op = _get_cache_prim(MaxPoolGradWithArgmax)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def max_pool_grad_with_argmax_v2(*args):
    op = _get_cache_prim(MaxPoolGradWithArgmaxV2)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def max_unpool2_d_grad(*args):
    op = _get_cache_prim(MaxUnpool2DGrad)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


def max_unpool3_d_grad(*args):
    op = _get_cache_prim(MaxUnpool3DGrad)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


def maximum_grad(*args):
    op = _get_cache_prim(MaximumGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def maximum_grad_grad(*args):
    op = _get_cache_prim(MaximumGradGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def median_grad(*args):
    op = _get_cache_prim(MedianGrad)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def minimum_grad(*args):
    op = _get_cache_prim(MinimumGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


minimum_grad_grad_op = MinimumGradGrad()
def minimum_grad_grad(*args):
    minimum_grad_grad_op = hook_call(minimum_grad_grad_op)
    return minimum_grad_grad_op(*args)


def mirror_pad_grad(*args):
    op = _get_cache_prim(MirrorPadGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def multi_margin_loss_grad(*args):
    op = _get_cache_prim(MultiMarginLossGrad)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def multilabel_margin_loss_grad(*args):
    op = _get_cache_prim(MultilabelMarginLossGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def mvlgamma_grad(*args):
    op = _get_cache_prim(MvlgammaGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def nll_loss_grad(*args):
    op = _get_cache_prim(NLLLossGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def neighbor_exchange_v2_grad(*args):
    op = _get_cache_prim(NeighborExchangeV2Grad)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


p_re_lu_grad_op = PReLUGrad()
def p_re_lu_grad(*args):
    p_re_lu_grad_op = hook_call(p_re_lu_grad_op)
    return p_re_lu_grad_op(*args)


def psroi_pooling_grad(*args):
    op = _get_cache_prim(PSROIPoolingGrad)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def pad_v3_grad(*args):
    op = _get_cache_prim(PadV3Grad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def parallel_resize_bilinear_grad(*args):
    op = _get_cache_prim(ParallelResizeBilinearGrad)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def pdist_grad(*args):
    op = _get_cache_prim(PdistGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def primitive(*args):
    op = _get_cache_prim(Primitive)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def primitive_with_infer(*args):
    op = _get_cache_prim(PrimitiveWithInfer)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def ps_roi_pooling_grad(*args):
    op = _get_cache_prim(PsROIPoolingGrad)(*args[-9:])
    op = hook_call(op)
    return op(*args[:-9])


def roi_align_grad(*args):
    op = _get_cache_prim(ROIAlignGrad)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


random_gamma_grad_op = RandomGammaGrad()
def random_gamma_grad(*args):
    random_gamma_grad_op = hook_call(random_gamma_grad_op)
    return random_gamma_grad_op(*args)


re_lu6_grad_op = ReLU6Grad()
def re_lu6_grad(*args):
    re_lu6_grad_op = hook_call(re_lu6_grad_op)
    return re_lu6_grad_op(*args)


reciprocal_grad_op = ReciprocalGrad()
def reciprocal_grad(*args):
    reciprocal_grad_op = hook_call(reciprocal_grad_op)
    return reciprocal_grad_op(*args)


ref_to_embed_op = RefToEmbed()
def ref_to_embed(*args):
    ref_to_embed_op = hook_call(ref_to_embed_op)
    return ref_to_embed_op(*args)


relu_grad_op = ReluGrad()
def relu_grad(*args):
    relu_grad_op = hook_call(relu_grad_op)
    return relu_grad_op(*args)


def resize_bicubic_grad(*args):
    op = _get_cache_prim(ResizeBicubicGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def resize_bilinear_grad(*args):
    op = _get_cache_prim(ResizeBilinearGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def resize_linear1_d_grad(*args):
    op = _get_cache_prim(ResizeLinear1DGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def resize_nearest_neighbor_grad(*args):
    op = _get_cache_prim(ResizeNearestNeighborGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def resize_nearest_neighbor_v2_grad(*args):
    op = _get_cache_prim(ResizeNearestNeighborV2Grad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def resize_v2_grad(*args):
    op = _get_cache_prim(ResizeV2Grad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


rms_norm_grad_op = RmsNormGrad()
def rms_norm_grad(*args):
    rms_norm_grad_op = hook_call(rms_norm_grad_op)
    return rms_norm_grad_op(*args)


rsqrt_grad_op = RsqrtGrad()
def rsqrt_grad(*args):
    rsqrt_grad_op = hook_call(rsqrt_grad_op)
    return rsqrt_grad_op(*args)


def scale_and_translate_grad(*args):
    op = _get_cache_prim(ScaleAndTranslateGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


selu_grad_op = SeluGrad()
def selu_grad(*args):
    selu_grad_op = hook_call(selu_grad_op)
    return selu_grad_op(*args)


si_lu_grad_op = SiLUGrad()
def si_lu_grad(*args):
    si_lu_grad_op = hook_call(si_lu_grad_op)
    return si_lu_grad_op(*args)


sigmoid_cross_entropy_with_logits_grad_op = SigmoidCrossEntropyWithLogitsGrad()
def sigmoid_cross_entropy_with_logits_grad(*args):
    sigmoid_cross_entropy_with_logits_grad_op = hook_call(sigmoid_cross_entropy_with_logits_grad_op)
    return sigmoid_cross_entropy_with_logits_grad_op(*args)


sigmoid_grad_op = SigmoidGrad()
def sigmoid_grad(*args):
    sigmoid_grad_op = hook_call(sigmoid_grad_op)
    return sigmoid_grad_op(*args)


slice_grad_op = SliceGrad()
def slice_grad(*args):
    slice_grad_op = hook_call(slice_grad_op)
    return slice_grad_op(*args)


def smooth_l1_loss_grad(*args):
    op = _get_cache_prim(SmoothL1LossGrad)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def soft_margin_loss_grad(*args):
    op = _get_cache_prim(SoftMarginLossGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def soft_shrink_grad(*args):
    op = _get_cache_prim(SoftShrinkGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


softmax_grad_op = SoftmaxGrad()
def softmax_grad(*args):
    softmax_grad_op = hook_call(softmax_grad_op)
    return softmax_grad_op(*args)


softplus_grad_op = SoftplusGrad()
def softplus_grad(*args):
    softplus_grad_op = hook_call(softplus_grad_op)
    return softplus_grad_op(*args)


sparse_fill_empty_rows_grad_op = SparseFillEmptyRowsGrad()
def sparse_fill_empty_rows_grad(*args):
    sparse_fill_empty_rows_grad_op = hook_call(sparse_fill_empty_rows_grad_op)
    return sparse_fill_empty_rows_grad_op(*args)


sparse_segment_mean_grad_op = SparseSegmentMeanGrad()
def sparse_segment_mean_grad(*args):
    sparse_segment_mean_grad_op = hook_call(sparse_segment_mean_grad_op)
    return sparse_segment_mean_grad_op(*args)


sparse_segment_sqrt_n_grad_op = SparseSegmentSqrtNGrad()
def sparse_segment_sqrt_n_grad(*args):
    sparse_segment_sqrt_n_grad_op = hook_call(sparse_segment_sqrt_n_grad_op)
    return sparse_segment_sqrt_n_grad_op(*args)


sparse_segment_sum_grad_op = SparseSegmentSumGrad()
def sparse_segment_sum_grad(*args):
    sparse_segment_sum_grad_op = hook_call(sparse_segment_sum_grad_op)
    return sparse_segment_sum_grad_op(*args)


sparse_slice_grad_op = SparseSliceGrad()
def sparse_slice_grad(*args):
    sparse_slice_grad_op = hook_call(sparse_slice_grad_op)
    return sparse_slice_grad_op(*args)


sqrt_grad_op = SqrtGrad()
def sqrt_grad(*args):
    sqrt_grad_op = hook_call(sqrt_grad_op)
    return sqrt_grad_op(*args)


def strided_slice_grad(*args):
    op = _get_cache_prim(StridedSliceGrad)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


def sync_batch_norm_grad(*args):
    op = _get_cache_prim(SyncBatchNormGrad)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


tanh_grad_op = TanhGrad()
def tanh_grad(*args):
    tanh_grad_op = hook_call(tanh_grad_op)
    return tanh_grad_op(*args)


trace_grad_op = TraceGrad()
def trace_grad(*args):
    trace_grad_op = hook_call(trace_grad_op)
    return trace_grad_op(*args)


unique_grad_op = UniqueGrad()
def unique_grad(*args):
    unique_grad_op = hook_call(unique_grad_op)
    return unique_grad_op(*args)


upsample_nearest3_d_grad_op = UpsampleNearest3DGrad()
def upsample_nearest3_d_grad(*args):
    upsample_nearest3_d_grad_op = hook_call(upsample_nearest3_d_grad_op)
    return upsample_nearest3_d_grad_op(*args)


def upsample_trilinear3_d_grad(*args):
    op = _get_cache_prim(UpsampleTrilinear3DGrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


wkv_grad_op = WKVGrad()
def wkv_grad(*args):
    wkv_grad_op = hook_call(wkv_grad_op)
    return wkv_grad_op(*args)


a_cos_op = ACos()
def a_cos(*args):
    a_cos_op = hook_call(a_cos_op)
    return a_cos_op(*args)


abs_op = Abs()
def abs(*args):
    abs_op = hook_call(abs_op)
    return abs_op(*args)


accumulate_nv2_op = AccumulateNV2()
def accumulate_nv2(*args):
    accumulate_nv2_op = hook_call(accumulate_nv2_op)
    return accumulate_nv2_op(*args)


acosh_op = Acosh()
def acosh(*args):
    acosh_op = hook_call(acosh_op)
    return acosh_op(*args)


def adam(*args):
    op = _get_cache_prim(Adam)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def adam_no_update_param(*args):
    op = _get_cache_prim(AdamNoUpdateParam)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def adam_weight_decay(*args):
    op = _get_cache_prim(AdamWeightDecay)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def adaptive_avg_pool2_d(*args):
    op = _get_cache_prim(AdaptiveAvgPool2D)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def adaptive_avg_pool3_d(*args):
    op = _get_cache_prim(AdaptiveAvgPool3D)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def adaptive_max_pool2_d(*args):
    op = _get_cache_prim(AdaptiveMaxPool2D)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


adaptive_max_pool3_d_op = AdaptiveMaxPool3D()
def adaptive_max_pool3_d(*args):
    adaptive_max_pool3_d_op = hook_call(adaptive_max_pool3_d_op)
    return adaptive_max_pool3_d_op(*args)


add_op = Add()
def add(*args):
    add_op = hook_call(add_op)
    return add_op(*args)


add_n_op = AddN()
def add_n(*args):
    add_n_op = hook_call(add_n_op)
    return add_n_op(*args)


addcdiv_op = Addcdiv()
def addcdiv(*args):
    addcdiv_op = hook_call(addcdiv_op)
    return addcdiv_op(*args)


addcmul_op = Addcmul()
def addcmul(*args):
    addcmul_op = hook_call(addcmul_op)
    return addcmul_op(*args)


adjust_hue_op = AdjustHue()
def adjust_hue(*args):
    adjust_hue_op = hook_call(adjust_hue_op)
    return adjust_hue_op(*args)


adjust_saturation_op = AdjustSaturation()
def adjust_saturation(*args):
    adjust_saturation_op = hook_call(adjust_saturation_op)
    return adjust_saturation_op(*args)


def affine_grid(*args):
    op = _get_cache_prim(AffineGrid)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def all_gather(*args):
    op = _get_cache_prim(AllGather)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def all_gather_v(*args):
    op = _get_cache_prim(AllGatherV)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def all_reduce(*args):
    op = _get_cache_prim(AllReduce)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def allto_all(*args):
    op = _get_cache_prim(AlltoAll)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def allto_all_v(*args):
    op = _get_cache_prim(AlltoAllV)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def allto_all_vc(*args):
    op = _get_cache_prim(AlltoAllVC)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


angle_op = Angle()
def angle(*args):
    angle_op = hook_call(angle_op)
    return angle_op(*args)


apply_ada_max_op = ApplyAdaMax()
def apply_ada_max(*args):
    apply_ada_max_op = hook_call(apply_ada_max_op)
    return apply_ada_max_op(*args)


apply_adadelta_op = ApplyAdadelta()
def apply_adadelta(*args):
    apply_adadelta_op = hook_call(apply_adadelta_op)
    return apply_adadelta_op(*args)


def apply_adagrad(*args):
    op = _get_cache_prim(ApplyAdagrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def apply_adagrad_da(*args):
    op = _get_cache_prim(ApplyAdagradDA)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def apply_adagrad_v2(*args):
    op = _get_cache_prim(ApplyAdagradV2)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def apply_adam_with_amsgrad(*args):
    op = _get_cache_prim(ApplyAdamWithAmsgrad)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def apply_adam_with_amsgrad_v2(*args):
    op = _get_cache_prim(ApplyAdamWithAmsgradV2)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


apply_add_sign_op = ApplyAddSign()
def apply_add_sign(*args):
    apply_add_sign_op = hook_call(apply_add_sign_op)
    return apply_add_sign_op(*args)


def apply_centered_rms_prop(*args):
    op = _get_cache_prim(ApplyCenteredRMSProp)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def apply_ftrl(*args):
    op = _get_cache_prim(ApplyFtrl)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


apply_gradient_descent_op = ApplyGradientDescent()
def apply_gradient_descent(*args):
    apply_gradient_descent_op = hook_call(apply_gradient_descent_op)
    return apply_gradient_descent_op(*args)


def apply_keras_momentum(*args):
    op = _get_cache_prim(ApplyKerasMomentum)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def apply_momentum(*args):
    op = _get_cache_prim(ApplyMomentum)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


apply_power_sign_op = ApplyPowerSign()
def apply_power_sign(*args):
    apply_power_sign_op = hook_call(apply_power_sign_op)
    return apply_power_sign_op(*args)


def apply_proximal_adagrad(*args):
    op = _get_cache_prim(ApplyProximalAdagrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


apply_proximal_gradient_descent_op = ApplyProximalGradientDescent()
def apply_proximal_gradient_descent(*args):
    apply_proximal_gradient_descent_op = hook_call(apply_proximal_gradient_descent_op)
    return apply_proximal_gradient_descent_op(*args)


def apply_rms_prop(*args):
    op = _get_cache_prim(ApplyRMSProp)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def apply_rotary_pos_emb(*args):
    op = _get_cache_prim(ApplyRotaryPosEmb)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def approximate_equal(*args):
    op = _get_cache_prim(ApproximateEqual)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def arg_max_with_value(*args):
    op = _get_cache_prim(ArgMaxWithValue)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def arg_min_with_value(*args):
    op = _get_cache_prim(ArgMinWithValue)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def argmax(*args):
    op = _get_cache_prim(Argmax)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def argmin(*args):
    op = _get_cache_prim(Argmin)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


asin_op = Asin()
def asin(*args):
    asin_op = hook_call(asin_op)
    return asin_op(*args)


asinh_op = Asinh()
def asinh(*args):
    asinh_op = hook_call(asinh_op)
    return asinh_op(*args)


assign_op = Assign()
def assign(*args):
    assign_op = hook_call(assign_op)
    return assign_op(*args)


assign_add_op = AssignAdd()
def assign_add(*args):
    assign_add_op = hook_call(assign_add_op)
    return assign_add_op(*args)


assign_sub_op = AssignSub()
def assign_sub(*args):
    assign_sub_op = hook_call(assign_sub_op)
    return assign_sub_op(*args)


atan_op = Atan()
def atan(*args):
    atan_op = hook_call(atan_op)
    return atan_op(*args)


atan2_op = Atan2()
def atan2(*args):
    atan2_op = hook_call(atan2_op)
    return atan2_op(*args)


atanh_op = Atanh()
def atanh(*args):
    atanh_op = hook_call(atanh_op)
    return atanh_op(*args)


def avg_pool(*args):
    op = _get_cache_prim(AvgPool)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def avg_pool3_d(*args):
    op = _get_cache_prim(AvgPool3D)(*args[-8:])
    op = hook_call(op)
    return op(*args[:-8])


def bce_with_logits_loss(*args):
    op = _get_cache_prim(BCEWithLogitsLoss)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def barrier(*args):
    op = _get_cache_prim(Barrier)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def bartlett_window(*args):
    op = _get_cache_prim(BartlettWindow)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def basic_lstm_cell(*args):
    op = _get_cache_prim(BasicLSTMCell)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def batch_i_send_i_recv(*args):
    op = _get_cache_prim(BatchISendIRecv)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


def batch_mat_mul(*args):
    op = _get_cache_prim(BatchMatMul)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def batch_norm(*args):
    op = _get_cache_prim(BatchNorm)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def batch_to_space(*args):
    op = _get_cache_prim(BatchToSpace)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def batch_to_space_nd(*args):
    op = _get_cache_prim(BatchToSpaceND)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


batch_to_space_ndv2_op = BatchToSpaceNDV2()
def batch_to_space_ndv2(*args):
    batch_to_space_ndv2_op = hook_call(batch_to_space_ndv2_op)
    return batch_to_space_ndv2_op(*args)


def bernoulli(*args):
    op = _get_cache_prim(Bernoulli)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


bessel_i0_op = BesselI0()
def bessel_i0(*args):
    bessel_i0_op = hook_call(bessel_i0_op)
    return bessel_i0_op(*args)


bessel_i0e_op = BesselI0e()
def bessel_i0e(*args):
    bessel_i0e_op = hook_call(bessel_i0e_op)
    return bessel_i0e_op(*args)


bessel_i1_op = BesselI1()
def bessel_i1(*args):
    bessel_i1_op = hook_call(bessel_i1_op)
    return bessel_i1_op(*args)


bessel_i1e_op = BesselI1e()
def bessel_i1e(*args):
    bessel_i1e_op = hook_call(bessel_i1e_op)
    return bessel_i1e_op(*args)


bessel_j0_op = BesselJ0()
def bessel_j0(*args):
    bessel_j0_op = hook_call(bessel_j0_op)
    return bessel_j0_op(*args)


bessel_j1_op = BesselJ1()
def bessel_j1(*args):
    bessel_j1_op = hook_call(bessel_j1_op)
    return bessel_j1_op(*args)


bessel_k0_op = BesselK0()
def bessel_k0(*args):
    bessel_k0_op = hook_call(bessel_k0_op)
    return bessel_k0_op(*args)


bessel_k0e_op = BesselK0e()
def bessel_k0e(*args):
    bessel_k0e_op = hook_call(bessel_k0e_op)
    return bessel_k0e_op(*args)


bessel_k1_op = BesselK1()
def bessel_k1(*args):
    bessel_k1_op = hook_call(bessel_k1_op)
    return bessel_k1_op(*args)


bessel_k1e_op = BesselK1e()
def bessel_k1e(*args):
    bessel_k1e_op = hook_call(bessel_k1e_op)
    return bessel_k1e_op(*args)


bessel_y0_op = BesselY0()
def bessel_y0(*args):
    bessel_y0_op = hook_call(bessel_y0_op)
    return bessel_y0_op(*args)


bessel_y1_op = BesselY1()
def bessel_y1(*args):
    bessel_y1_op = hook_call(bessel_y1_op)
    return bessel_y1_op(*args)


betainc_op = Betainc()
def betainc(*args):
    betainc_op = hook_call(betainc_op)
    return betainc_op(*args)


def bias_add(*args):
    op = _get_cache_prim(BiasAdd)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def binary_cross_entropy(*args):
    op = _get_cache_prim(BinaryCrossEntropy)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


bincount_op = Bincount()
def bincount(*args):
    bincount_op = hook_call(bincount_op)
    return bincount_op(*args)


bitwise_and_op = BitwiseAnd()
def bitwise_and(*args):
    bitwise_and_op = hook_call(bitwise_and_op)
    return bitwise_and_op(*args)


bitwise_or_op = BitwiseOr()
def bitwise_or(*args):
    bitwise_or_op = hook_call(bitwise_or_op)
    return bitwise_or_op(*args)


bitwise_xor_op = BitwiseXor()
def bitwise_xor(*args):
    bitwise_xor_op = hook_call(bitwise_xor_op)
    return bitwise_xor_op(*args)


def blackman_window(*args):
    op = _get_cache_prim(BlackmanWindow)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def bounding_box_decode(*args):
    op = _get_cache_prim(BoundingBoxDecode)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def bounding_box_encode(*args):
    op = _get_cache_prim(BoundingBoxEncode)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def broadcast(*args):
    op = _get_cache_prim(Broadcast)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def broadcast_to(*args):
    op = _get_cache_prim(BroadcastTo)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def bucketize(*args):
    op = _get_cache_prim(Bucketize)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def ctc_greedy_decoder(*args):
    op = _get_cache_prim(CTCGreedyDecoder)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def ctc_loss(*args):
    op = _get_cache_prim(CTCLoss)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def ctc_loss_v2(*args):
    op = _get_cache_prim(CTCLossV2)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


cast_op = Cast()
def cast(*args):
    cast_op = hook_call(cast_op)
    return cast_op(*args)


def cauchy(*args):
    op = _get_cache_prim(Cauchy)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def cdist(*args):
    op = _get_cache_prim(Cdist)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def ce_lu(*args):
    op = _get_cache_prim(CeLU)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


ceil_op = Ceil()
def ceil(*args):
    ceil_op = hook_call(ceil_op)
    return ceil_op(*args)


def channel_shuffle(*args):
    op = _get_cache_prim(ChannelShuffle)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


check_numerics_op = CheckNumerics()
def check_numerics(*args):
    check_numerics_op = hook_call(check_numerics_op)
    return check_numerics_op(*args)


check_valid_op = CheckValid()
def check_valid(*args):
    check_valid_op = hook_call(check_valid_op)
    return check_valid_op(*args)


def cholesky(*args):
    op = _get_cache_prim(Cholesky)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def cholesky_inverse(*args):
    op = _get_cache_prim(CholeskyInverse)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def cholesky_solve(*args):
    op = _get_cache_prim(CholeskySolve)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


coalesce_op = Coalesce()
def coalesce(*args):
    coalesce_op = hook_call(coalesce_op)
    return coalesce_op(*args)


def col2_im(*args):
    op = _get_cache_prim(Col2Im)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def collective_gather(*args):
    op = _get_cache_prim(CollectiveGather)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def collective_scatter(*args):
    op = _get_cache_prim(CollectiveScatter)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def combined_non_max_suppression(*args):
    op = _get_cache_prim(CombinedNonMaxSuppression)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


compare_and_bitpack_op = CompareAndBitpack()
def compare_and_bitpack(*args):
    compare_and_bitpack_op = hook_call(compare_and_bitpack_op)
    return compare_and_bitpack_op(*args)


complex_op = Complex()
def complex(*args):
    complex_op = hook_call(complex_op)
    return complex_op(*args)


complex_abs_op = ComplexAbs()
def complex_abs(*args):
    complex_abs_op = hook_call(complex_abs_op)
    return complex_abs_op(*args)


def compute_accidental_hits(*args):
    op = _get_cache_prim(ComputeAccidentalHits)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def concat(*args):
    op = _get_cache_prim(Concat)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def confusion_matrix(*args):
    op = _get_cache_prim(ConfusionMatrix)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


conj_op = Conj()
def conj(*args):
    conj_op = hook_call(conj_op)
    return conj_op(*args)


conjugate_transpose_op = ConjugateTranspose()
def conjugate_transpose(*args):
    conjugate_transpose_op = hook_call(conjugate_transpose_op)
    return conjugate_transpose_op(*args)


def conv2_d(*args):
    op = _get_cache_prim(Conv2D)(*args[-9:])
    op = hook_call(op)
    return op(*args[:-9])


def conv2_d_backprop_input(*args):
    op = _get_cache_prim(Conv2DBackpropInput)(*args[-10:])
    op = hook_call(op)
    return op(*args[:-10])


def conv2_d_transpose(*args):
    op = _get_cache_prim(Conv2DTranspose)(*args[-10:])
    op = hook_call(op)
    return op(*args[:-10])


def conv3_d(*args):
    op = _get_cache_prim(Conv3D)(*args[-9:])
    op = hook_call(op)
    return op(*args[:-9])


def conv3_d_transpose(*args):
    op = _get_cache_prim(Conv3DTranspose)(*args[-11:])
    op = hook_call(op)
    return op(*args[:-11])


copy_with_slice_op = CopyWithSlice()
def copy_with_slice(*args):
    copy_with_slice_op = hook_call(copy_with_slice_op)
    return copy_with_slice_op(*args)


cos_op = Cos()
def cos(*args):
    cos_op = hook_call(cos_op)
    return cos_op(*args)


cosh_op = Cosh()
def cosh(*args):
    cosh_op = hook_call(cosh_op)
    return cosh_op(*args)


def count_non_zero(*args):
    op = _get_cache_prim(CountNonZero)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def crop_and_resize(*args):
    op = _get_cache_prim(CropAndResize)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def cross(*args):
    op = _get_cache_prim(Cross)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def cum_prod(*args):
    op = _get_cache_prim(CumProd)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def cum_sum(*args):
    op = _get_cache_prim(CumSum)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def cummax(*args):
    op = _get_cache_prim(Cummax)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def cummin(*args):
    op = _get_cache_prim(Cummin)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def cumulative_logsumexp(*args):
    op = _get_cache_prim(CumulativeLogsumexp)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


d_type_op = DType()
def d_type(*args):
    d_type_op = hook_call(d_type_op)
    return d_type_op(*args)


def data_format_dim_map(*args):
    op = _get_cache_prim(DataFormatDimMap)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def data_format_vec_permute(*args):
    op = _get_cache_prim(DataFormatVecPermute)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def deformable_offsets(*args):
    op = _get_cache_prim(DeformableOffsets)(*args[-7:])
    op = hook_call(op)
    return op(*args[:-7])


dense_op = Dense()
def dense(*args):
    dense_op = hook_call(dense_op)
    return dense_op(*args)


depend_op = Depend()
def depend(*args):
    depend_op = hook_call(depend_op)
    return depend_op(*args)


def depth_to_space(*args):
    op = _get_cache_prim(DepthToSpace)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def depthwise_conv2d_native(*args):
    op = _get_cache_prim(DepthwiseConv2dNative)(*args[-8:])
    op = hook_call(op)
    return op(*args[:-8])


diag_op = Diag()
def diag(*args):
    diag_op = hook_call(diag_op)
    return diag_op(*args)


diag_part_op = DiagPart()
def diag_part(*args):
    diag_part_op = hook_call(diag_part_op)
    return diag_part_op(*args)


digamma_op = Digamma()
def digamma(*args):
    digamma_op = hook_call(digamma_op)
    return digamma_op(*args)


def dilation2_d(*args):
    op = _get_cache_prim(Dilation2D)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


div_op = Div()
def div(*args):
    div_op = hook_call(div_op)
    return div_op(*args)


div_no_nan_op = DivNoNan()
def div_no_nan(*args):
    div_no_nan_op = hook_call(div_no_nan_op)
    return div_no_nan_op(*args)


def dropout(*args):
    op = _get_cache_prim(Dropout)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def dropout2_d(*args):
    op = _get_cache_prim(Dropout2D)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def dropout3_d(*args):
    op = _get_cache_prim(Dropout3D)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


dropout_do_mask_op = DropoutDoMask()
def dropout_do_mask(*args):
    dropout_do_mask_op = hook_call(dropout_do_mask_op)
    return dropout_do_mask_op(*args)


def dropout_gen_mask(*args):
    op = _get_cache_prim(DropoutGenMask)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


dump_gradient_op = DumpGradient()
def dump_gradient(*args):
    dump_gradient_op = hook_call(dump_gradient_op)
    return dump_gradient_op(*args)


def dynamic_gruv2(*args):
    op = _get_cache_prim(DynamicGRUV2)(*args[-10:])
    op = hook_call(op)
    return op(*args[:-10])


def dynamic_rnn(*args):
    op = _get_cache_prim(DynamicRNN)(*args[-11:])
    op = hook_call(op)
    return op(*args[:-11])


def dynamic_shape(*args):
    op = _get_cache_prim(DynamicShape)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def edit_distance(*args):
    op = _get_cache_prim(EditDistance)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def eig(*args):
    op = _get_cache_prim(Eig)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def einsum(*args):
    op = _get_cache_prim(Einsum)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def elu(*args):
    op = _get_cache_prim(Elu)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


embedding_lookup_op = EmbeddingLookup()
def embedding_lookup(*args):
    embedding_lookup_op = hook_call(embedding_lookup_op)
    return embedding_lookup_op(*args)


eps_op = Eps()
def eps(*args):
    eps_op = hook_call(eps_op)
    return eps_op(*args)


equal_op = Equal()
def equal(*args):
    equal_op = hook_call(equal_op)
    return equal_op(*args)


equal_count_op = EqualCount()
def equal_count(*args):
    equal_count_op = hook_call(equal_count_op)
    return equal_count_op(*args)


erf_op = Erf()
def erf(*args):
    erf_op = hook_call(erf_op)
    return erf_op(*args)


erfc_op = Erfc()
def erfc(*args):
    erfc_op = hook_call(erfc_op)
    return erfc_op(*args)


erfinv_op = Erfinv()
def erfinv(*args):
    erfinv_op = hook_call(erfinv_op)
    return erfinv_op(*args)


erfinv_op = Erfinv()
def erfinv(*args):
    erfinv_op = hook_call(erfinv_op)
    return erfinv_op(*args)


def euclidean_norm(*args):
    op = _get_cache_prim(EuclideanNorm)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


exp_op = Exp()
def exp(*args):
    exp_op = hook_call(exp_op)
    return exp_op(*args)


expand_op = Expand()
def expand(*args):
    expand_op = hook_call(expand_op)
    return expand_op(*args)


expand_dims_op = ExpandDims()
def expand_dims(*args):
    expand_dims_op = hook_call(expand_dims_op)
    return expand_dims_op(*args)


expm1_op = Expm1()
def expm1(*args):
    expm1_op = hook_call(expm1_op)
    return expm1_op(*args)


def extract_glimpse(*args):
    op = _get_cache_prim(ExtractGlimpse)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def extract_image_patches(*args):
    op = _get_cache_prim(ExtractImagePatches)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def extract_volume_patches(*args):
    op = _get_cache_prim(ExtractVolumePatches)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


eye_op = Eye()
def eye(*args):
    eye_op = hook_call(eye_op)
    return eye_op(*args)


def fft_with_size(*args):
    op = _get_cache_prim(FFTWithSize)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


fast_ge_lu_op = FastGeLU()
def fast_ge_lu(*args):
    fast_ge_lu_op = hook_call(fast_ge_lu_op)
    return fast_ge_lu_op(*args)


fast_gelu_op = FastGelu()
def fast_gelu(*args):
    fast_gelu_op = hook_call(fast_gelu_op)
    return fast_gelu_op(*args)


fill_op = Fill()
def fill(*args):
    fill_op = hook_call(fill_op)
    return fill_op(*args)


def fill_diagonal(*args):
    op = _get_cache_prim(FillDiagonal)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


fill_v2_op = FillV2()
def fill_v2(*args):
    fill_v2_op = hook_call(fill_v2_op)
    return fill_v2_op(*args)


fills_op = Fills()
def fills(*args):
    fills_op = hook_call(fills_op)
    return fills_op(*args)


flatten_op = Flatten()
def flatten(*args):
    flatten_op = hook_call(flatten_op)
    return flatten_op(*args)


float_status_op = FloatStatus()
def float_status(*args):
    float_status_op = hook_call(float_status_op)
    return float_status_op(*args)


floor_op = Floor()
def floor(*args):
    floor_op = hook_call(floor_op)
    return floor_op(*args)


floor_div_op = FloorDiv()
def floor_div(*args):
    floor_div_op = hook_call(floor_div_op)
    return floor_div_op(*args)


floor_mod_op = FloorMod()
def floor_mod(*args):
    floor_mod_op = hook_call(floor_mod_op)
    return floor_mod_op(*args)


fmax_op = Fmax()
def fmax(*args):
    fmax_op = hook_call(fmax_op)
    return fmax_op(*args)


fmin_op = Fmin()
def fmin(*args):
    fmin_op = hook_call(fmin_op)
    return fmin_op(*args)


fori_loop_op = ForiLoop()
def fori_loop(*args):
    fori_loop_op = hook_call(fori_loop_op)
    return fori_loop_op(*args)


def fractional_avg_pool(*args):
    op = _get_cache_prim(FractionalAvgPool)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def fractional_max_pool(*args):
    op = _get_cache_prim(FractionalMaxPool)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def fractional_max_pool3_d_with_fixed_ksize(*args):
    op = _get_cache_prim(FractionalMaxPool3DWithFixedKsize)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def fractional_max_pool_with_fixed_ksize(*args):
    op = _get_cache_prim(FractionalMaxPoolWithFixedKsize)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def fused_ada_factor(*args):
    op = _get_cache_prim(FusedAdaFactor)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def fused_ada_factor_with_global_norm(*args):
    op = _get_cache_prim(FusedAdaFactorWithGlobalNorm)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def fused_cast_adam_weight_decay(*args):
    op = _get_cache_prim(FusedCastAdamWeightDecay)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def fused_sparse_adam(*args):
    op = _get_cache_prim(FusedSparseAdam)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def fused_sparse_ftrl(*args):
    op = _get_cache_prim(FusedSparseFtrl)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


def fused_sparse_lazy_adam(*args):
    op = _get_cache_prim(FusedSparseLazyAdam)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def fused_sparse_proximal_adagrad(*args):
    op = _get_cache_prim(FusedSparseProximalAdagrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


fused_weight_scale_apply_momentum_op = FusedWeightScaleApplyMomentum()
def fused_weight_scale_apply_momentum(*args):
    fused_weight_scale_apply_momentum_op = hook_call(fused_weight_scale_apply_momentum_op)
    return fused_weight_scale_apply_momentum_op(*args)


def glu(*args):
    op = _get_cache_prim(GLU)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def gamma(*args):
    op = _get_cache_prim(Gamma)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def gather(*args):
    op = _get_cache_prim(Gather)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


gather_d_op = GatherD()
def gather_d(*args):
    gather_d_op = hook_call(gather_d_op)
    return gather_d_op(*args)


gather_nd_op = GatherNd()
def gather_nd(*args):
    gather_nd_op = hook_call(gather_nd_op)
    return gather_nd_op(*args)


gather_v2_op = GatherV2()
def gather_v2(*args):
    gather_v2_op = hook_call(gather_v2_op)
    return gather_v2_op(*args)


gcd_op = Gcd()
def gcd(*args):
    gcd_op = hook_call(gcd_op)
    return gcd_op(*args)


ge_lu_op = GeLU()
def ge_lu(*args):
    ge_lu_op = hook_call(ge_lu_op)
    return ge_lu_op(*args)


ge_switch_op = GeSwitch()
def ge_switch(*args):
    ge_switch_op = hook_call(ge_switch_op)
    return ge_switch_op(*args)


gelu_op = Gelu()
def gelu(*args):
    gelu_op = hook_call(gelu_op)
    return gelu_op(*args)


geqrf_op = Geqrf()
def geqrf(*args):
    geqrf_op = hook_call(geqrf_op)
    return geqrf_op(*args)


ger_op = Ger()
def ger(*args):
    ger_op = hook_call(ger_op)
    return ger_op(*args)


def get_next(*args):
    op = _get_cache_prim(GetNext)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


greater_op = Greater()
def greater(*args):
    greater_op = hook_call(greater_op)
    return greater_op(*args)


greater_equal_op = GreaterEqual()
def greater_equal(*args):
    greater_equal_op = hook_call(greater_equal_op)
    return greater_equal_op(*args)


def grid_sampler2_d(*args):
    op = _get_cache_prim(GridSampler2D)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def grid_sampler3_d(*args):
    op = _get_cache_prim(GridSampler3D)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


group_topk_op = GroupTopk()
def group_topk(*args):
    group_topk_op = hook_call(group_topk_op)
    return group_topk_op(*args)


hsv_to_rgb_op = HSVToRGB()
def hsv_to_rgb(*args):
    hsv_to_rgb_op = hook_call(hsv_to_rgb_op)
    return hsv_to_rgb_op(*args)


def h_shrink(*args):
    op = _get_cache_prim(HShrink)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


h_sigmoid_op = HSigmoid()
def h_sigmoid(*args):
    h_sigmoid_op = hook_call(h_sigmoid_op)
    return h_sigmoid_op(*args)


h_swish_op = HSwish()
def h_swish(*args):
    h_swish_op = hook_call(h_swish_op)
    return h_swish_op(*args)


def hamming_window(*args):
    op = _get_cache_prim(HammingWindow)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


heaviside_op = Heaviside()
def heaviside(*args):
    heaviside_op = hook_call(heaviside_op)
    return heaviside_op(*args)


def histogram(*args):
    op = _get_cache_prim(Histogram)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def histogram_fixed_width(*args):
    op = _get_cache_prim(HistogramFixedWidth)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


histogram_summary_op = HistogramSummary()
def histogram_summary(*args):
    histogram_summary_op = hook_call(histogram_summary_op)
    return histogram_summary_op(*args)


def hook_backward(*args):
    op = _get_cache_prim(HookBackward)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


hypot_op = Hypot()
def hypot(*args):
    hypot_op = hook_call(hypot_op)
    return hypot_op(*args)


def iou(*args):
    op = _get_cache_prim(IOU)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


identity_op = Identity()
def identity(*args):
    identity_op = hook_call(identity_op)
    return identity_op(*args)


identity_n_op = IdentityN()
def identity_n(*args):
    identity_n_op = hook_call(identity_n_op)
    return identity_n_op(*args)


igamma_op = Igamma()
def igamma(*args):
    igamma_op = hook_call(igamma_op)
    return igamma_op(*args)


igammac_op = Igammac()
def igammac(*args):
    igammac_op = hook_call(igammac_op)
    return igammac_op(*args)


def im2_col(*args):
    op = _get_cache_prim(Im2Col)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


imag_op = Imag()
def imag(*args):
    imag_op = hook_call(imag_op)
    return imag_op(*args)


image_summary_op = ImageSummary()
def image_summary(*args):
    image_summary_op = hook_call(image_summary_op)
    return image_summary_op(*args)


def in_top_k(*args):
    op = _get_cache_prim(InTopK)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def index_add(*args):
    op = _get_cache_prim(IndexAdd)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


index_fill_op = IndexFill()
def index_fill(*args):
    index_fill_op = hook_call(index_fill_op)
    return index_fill_op(*args)


def index_put(*args):
    op = _get_cache_prim(IndexPut)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def inplace_add(*args):
    op = _get_cache_prim(InplaceAdd)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def inplace_index_add(*args):
    op = _get_cache_prim(InplaceIndexAdd)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def inplace_sub(*args):
    op = _get_cache_prim(InplaceSub)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def inplace_update(*args):
    op = _get_cache_prim(InplaceUpdate)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


inplace_update_v2_op = InplaceUpdateV2()
def inplace_update_v2(*args):
    inplace_update_v2_op = hook_call(inplace_update_v2_op)
    return inplace_update_v2_op(*args)


def insert_gradient_of(*args):
    op = _get_cache_prim(InsertGradientOf)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


inv_op = Inv()
def inv(*args):
    inv_op = hook_call(inv_op)
    return inv_op(*args)


invert_op = Invert()
def invert(*args):
    invert_op = hook_call(invert_op)
    return invert_op(*args)


invert_permutation_op = InvertPermutation()
def invert_permutation(*args):
    invert_permutation_op = hook_call(invert_permutation_op)
    return invert_permutation_op(*args)


def is_close(*args):
    op = _get_cache_prim(IsClose)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


is_finite_op = IsFinite()
def is_finite(*args):
    is_finite_op = hook_call(is_finite_op)
    return is_finite_op(*args)


is_inf_op = IsInf()
def is_inf(*args):
    is_inf_op = hook_call(is_inf_op)
    return is_inf_op(*args)


is_nan_op = IsNan()
def is_nan(*args):
    is_nan_op = hook_call(is_nan_op)
    return is_nan_op(*args)


def kl_div_loss(*args):
    op = _get_cache_prim(KLDivLoss)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


l2_loss_op = L2Loss()
def l2_loss(*args):
    l2_loss_op = hook_call(l2_loss_op)
    return l2_loss_op(*args)


def l2_normalize(*args):
    op = _get_cache_prim(L2Normalize)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def lars_update(*args):
    op = _get_cache_prim(LARSUpdate)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def lrn(*args):
    op = _get_cache_prim(LRN)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


def lstm(*args):
    op = _get_cache_prim(LSTM)(*args[-7:])
    op = hook_call(op)
    return op(*args[:-7])


def layer_norm(*args):
    op = _get_cache_prim(LayerNorm)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


lcm_op = Lcm()
def lcm(*args):
    lcm_op = hook_call(lcm_op)
    return lcm_op(*args)


left_shift_op = LeftShift()
def left_shift(*args):
    left_shift_op = hook_call(left_shift_op)
    return left_shift_op(*args)


lerp_op = Lerp()
def lerp(*args):
    lerp_op = hook_call(lerp_op)
    return lerp_op(*args)


lerp_scalar_op = LerpScalar()
def lerp_scalar(*args):
    lerp_scalar_op = hook_call(lerp_scalar_op)
    return lerp_scalar_op(*args)


less_op = Less()
def less(*args):
    less_op = hook_call(less_op)
    return less_op(*args)


less_equal_op = LessEqual()
def less_equal(*args):
    less_equal_op = hook_call(less_equal_op)
    return less_equal_op(*args)


lgamma_op = Lgamma()
def lgamma(*args):
    lgamma_op = hook_call(lgamma_op)
    return lgamma_op(*args)


lin_space_op = LinSpace()
def lin_space(*args):
    lin_space_op = hook_call(lin_space_op)
    return lin_space_op(*args)


def list_diff(*args):
    op = _get_cache_prim(ListDiff)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


log_op = Log()
def log(*args):
    log_op = hook_call(log_op)
    return log_op(*args)


log1p_op = Log1p()
def log1p(*args):
    log1p_op = hook_call(log1p_op)
    return log1p_op(*args)


log_matrix_determinant_op = LogMatrixDeterminant()
def log_matrix_determinant(*args):
    log_matrix_determinant_op = hook_call(log_matrix_determinant_op)
    return log_matrix_determinant_op(*args)


def log_normal_reverse(*args):
    op = _get_cache_prim(LogNormalReverse)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def log_softmax(*args):
    op = _get_cache_prim(LogSoftmax)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


log_softmax_ext_op = LogSoftmaxExt()
def log_softmax_ext(*args):
    log_softmax_ext_op = hook_call(log_softmax_ext_op)
    return log_softmax_ext_op(*args)


def log_space(*args):
    op = _get_cache_prim(LogSpace)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def log_uniform_candidate_sampler(*args):
    op = _get_cache_prim(LogUniformCandidateSampler)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


logical_and_op = LogicalAnd()
def logical_and(*args):
    logical_and_op = hook_call(logical_and_op)
    return logical_and_op(*args)


logical_not_op = LogicalNot()
def logical_not(*args):
    logical_not_op = hook_call(logical_not_op)
    return logical_not_op(*args)


logical_or_op = LogicalOr()
def logical_or(*args):
    logical_or_op = hook_call(logical_or_op)
    return logical_or_op(*args)


logical_xor_op = LogicalXor()
def logical_xor(*args):
    logical_xor_op = hook_call(logical_xor_op)
    return logical_xor_op(*args)


def logit(*args):
    op = _get_cache_prim(Logit)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def lower_bound(*args):
    op = _get_cache_prim(LowerBound)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def lp_norm(*args):
    op = _get_cache_prim(LpNorm)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def lstsq(*args):
    op = _get_cache_prim(Lstsq)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


lu_solve_op = LuSolve()
def lu_solve(*args):
    lu_solve_op = hook_call(lu_solve_op)
    return lu_solve_op(*args)


def lu_unpack(*args):
    op = _get_cache_prim(LuUnpack)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


map_cache_idx_op = MapCacheIdx()
def map_cache_idx(*args):
    map_cache_idx_op = hook_call(map_cache_idx_op)
    return map_cache_idx_op(*args)


map_uniform_op = MapUniform()
def map_uniform(*args):
    map_uniform_op = hook_call(map_uniform_op)
    return map_uniform_op(*args)


masked_fill_op = MaskedFill()
def masked_fill(*args):
    masked_fill_op = hook_call(masked_fill_op)
    return masked_fill_op(*args)


masked_scatter_op = MaskedScatter()
def masked_scatter(*args):
    masked_scatter_op = hook_call(masked_scatter_op)
    return masked_scatter_op(*args)


masked_select_op = MaskedSelect()
def masked_select(*args):
    masked_select_op = hook_call(masked_select_op)
    return masked_select_op(*args)


def mat_mul(*args):
    op = _get_cache_prim(MatMul)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


matrix_band_part_op = MatrixBandPart()
def matrix_band_part(*args):
    matrix_band_part_op = hook_call(matrix_band_part_op)
    return matrix_band_part_op(*args)


matrix_determinant_op = MatrixDeterminant()
def matrix_determinant(*args):
    matrix_determinant_op = hook_call(matrix_determinant_op)
    return matrix_determinant_op(*args)


def matrix_diag_part_v3(*args):
    op = _get_cache_prim(MatrixDiagPartV3)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def matrix_diag_v3(*args):
    op = _get_cache_prim(MatrixDiagV3)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


matrix_exp_op = MatrixExp()
def matrix_exp(*args):
    matrix_exp_op = hook_call(matrix_exp_op)
    return matrix_exp_op(*args)


def matrix_inverse(*args):
    op = _get_cache_prim(MatrixInverse)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


matrix_logarithm_op = MatrixLogarithm()
def matrix_logarithm(*args):
    matrix_logarithm_op = hook_call(matrix_logarithm_op)
    return matrix_logarithm_op(*args)


def matrix_power(*args):
    op = _get_cache_prim(MatrixPower)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def matrix_set_diag_v3(*args):
    op = _get_cache_prim(MatrixSetDiagV3)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def matrix_solve(*args):
    op = _get_cache_prim(MatrixSolve)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def matrix_solve_ls(*args):
    op = _get_cache_prim(MatrixSolveLs)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def matrix_triangular_solve(*args):
    op = _get_cache_prim(MatrixTriangularSolve)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def max_pool(*args):
    op = _get_cache_prim(MaxPool)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def max_pool3_d(*args):
    op = _get_cache_prim(MaxPool3D)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def max_pool3_d_with_argmax(*args):
    op = _get_cache_prim(MaxPool3DWithArgmax)(*args[-7:])
    op = hook_call(op)
    return op(*args[:-7])


def max_pool_with_argmax(*args):
    op = _get_cache_prim(MaxPoolWithArgmax)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def max_pool_with_argmax_v2(*args):
    op = _get_cache_prim(MaxPoolWithArgmaxV2)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def max_unpool2_d(*args):
    op = _get_cache_prim(MaxUnpool2D)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


def max_unpool3_d(*args):
    op = _get_cache_prim(MaxUnpool3D)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


maximum_op = Maximum()
def maximum(*args):
    maximum_op = hook_call(maximum_op)
    return maximum_op(*args)


merge_op = Merge()
def merge(*args):
    merge_op = hook_call(merge_op)
    return merge_op(*args)


def meshgrid(*args):
    op = _get_cache_prim(Meshgrid)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


minimum_op = Minimum()
def minimum(*args):
    minimum_op = hook_call(minimum_op)
    return minimum_op(*args)


def mirror_pad(*args):
    op = _get_cache_prim(MirrorPad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


mish_op = Mish()
def mish(*args):
    mish_op = hook_call(mish_op)
    return mish_op(*args)


mod_op = Mod()
def mod(*args):
    mod_op = hook_call(mod_op)
    return mod_op(*args)


def morph(*args):
    op = _get_cache_prim(Morph)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


move_to_op = MoveTo()
def move_to(*args):
    move_to_op = hook_call(move_to_op)
    return move_to_op(*args)


mul_op = Mul()
def mul(*args):
    mul_op = hook_call(mul_op)
    return mul_op(*args)


mul_no_nan_op = MulNoNan()
def mul_no_nan(*args):
    mul_no_nan_op = hook_call(mul_no_nan_op)
    return mul_no_nan_op(*args)


def multi_margin_loss(*args):
    op = _get_cache_prim(MultiMarginLoss)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def multilabel_margin_loss(*args):
    op = _get_cache_prim(MultilabelMarginLoss)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def multinomial(*args):
    op = _get_cache_prim(Multinomial)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def multinomial_with_replacement(*args):
    op = _get_cache_prim(MultinomialWithReplacement)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def mvlgamma(*args):
    op = _get_cache_prim(Mvlgamma)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def nll_loss(*args):
    op = _get_cache_prim(NLLLoss)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def nms_with_mask(*args):
    op = _get_cache_prim(NMSWithMask)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


npu_alloc_float_status_op = NPUAllocFloatStatus()
def npu_alloc_float_status(*args):
    npu_alloc_float_status_op = hook_call(npu_alloc_float_status_op)
    return npu_alloc_float_status_op(*args)


npu_clear_float_status_op = NPUClearFloatStatus()
def npu_clear_float_status(*args):
    npu_clear_float_status_op = hook_call(npu_clear_float_status_op)
    return npu_clear_float_status_op(*args)


npu_get_float_status_op = NPUGetFloatStatus()
def npu_get_float_status(*args):
    npu_get_float_status_op = hook_call(npu_get_float_status_op)
    return npu_get_float_status_op(*args)


def nan_to_num(*args):
    op = _get_cache_prim(NanToNum)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


neg_op = Neg()
def neg(*args):
    neg_op = hook_call(neg_op)
    return neg_op(*args)


def neighbor_exchange(*args):
    op = _get_cache_prim(NeighborExchange)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def neighbor_exchange_v2(*args):
    op = _get_cache_prim(NeighborExchangeV2)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


next_after_op = NextAfter()
def next_after(*args):
    next_after_op = hook_call(next_after_op)
    return next_after_op(*args)


def no_repeat_n_gram(*args):
    op = _get_cache_prim(NoRepeatNGram)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def non_deterministic_ints(*args):
    op = _get_cache_prim(NonDeterministicInts)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


non_max_suppression_v3_op = NonMaxSuppressionV3()
def non_max_suppression_v3(*args):
    non_max_suppression_v3_op = hook_call(non_max_suppression_v3_op)
    return non_max_suppression_v3_op(*args)


non_max_suppression_with_overlaps_op = NonMaxSuppressionWithOverlaps()
def non_max_suppression_with_overlaps(*args):
    non_max_suppression_with_overlaps_op = hook_call(non_max_suppression_with_overlaps_op)
    return non_max_suppression_with_overlaps_op(*args)


non_zero_op = NonZero()
def non_zero(*args):
    non_zero_op = hook_call(non_zero_op)
    return non_zero_op(*args)


not_equal_op = NotEqual()
def not_equal(*args):
    not_equal_op = hook_call(not_equal_op)
    return not_equal_op(*args)


def nth_element(*args):
    op = _get_cache_prim(NthElement)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def nuclear_norm(*args):
    op = _get_cache_prim(NuclearNorm)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def one_hot(*args):
    op = _get_cache_prim(OneHot)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


ones_op = Ones()
def ones(*args):
    ones_op = hook_call(ones_op)
    return ones_op(*args)


ones_like_op = OnesLike()
def ones_like(*args):
    ones_like_op = hook_call(ones_like_op)
    return ones_like_op(*args)


orgqr_op = Orgqr()
def orgqr(*args):
    orgqr_op = hook_call(orgqr_op)
    return orgqr_op(*args)


def ormqr(*args):
    op = _get_cache_prim(Ormqr)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


p_re_lu_op = PReLU()
def p_re_lu(*args):
    p_re_lu_op = hook_call(p_re_lu_op)
    return p_re_lu_op(*args)


def pack(*args):
    op = _get_cache_prim(Pack)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def pad(*args):
    op = _get_cache_prim(Pad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def pad_v3(*args):
    op = _get_cache_prim(PadV3)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def padding(*args):
    op = _get_cache_prim(Padding)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def paged_attention(*args):
    op = _get_cache_prim(PagedAttention)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def paged_attention_mask(*args):
    op = _get_cache_prim(PagedAttentionMask)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


parallel_concat_op = ParallelConcat()
def parallel_concat(*args):
    parallel_concat_op = hook_call(parallel_concat_op)
    return parallel_concat_op(*args)


def parameterized_truncated_normal(*args):
    op = _get_cache_prim(ParameterizedTruncatedNormal)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


partial_op = Partial()
def partial(*args):
    partial_op = hook_call(partial_op)
    return partial_op(*args)


def pdist(*args):
    op = _get_cache_prim(Pdist)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def poisson(*args):
    op = _get_cache_prim(Poisson)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


polar_op = Polar()
def polar(*args):
    polar_op = hook_call(polar_op)
    return polar_op(*args)


polygamma_op = Polygamma()
def polygamma(*args):
    polygamma_op = hook_call(polygamma_op)
    return polygamma_op(*args)


population_count_op = PopulationCount()
def population_count(*args):
    population_count_op = hook_call(population_count_op)
    return population_count_op(*args)


pow_op = Pow()
def pow(*args):
    pow_op = hook_call(pow_op)
    return pow_op(*args)


pull_op = Pull()
def pull(*args):
    pull_op = hook_call(pull_op)
    return pull_op(*args)


def push(*args):
    op = _get_cache_prim(Push)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


py_execute_op = PyExecute()
def py_execute(*args):
    py_execute_op = hook_call(py_execute_op)
    return py_execute_op(*args)


def py_func(*args):
    op = _get_cache_prim(PyFunc)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def qr(*args):
    op = _get_cache_prim(Qr)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def quantile(*args):
    op = _get_cache_prim(Quantile)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


rgb_to_hsv_op = RGBToHSV()
def rgb_to_hsv(*args):
    rgb_to_hsv_op = hook_call(rgb_to_hsv_op)
    return rgb_to_hsv_op(*args)


def rnnt_loss(*args):
    op = _get_cache_prim(RNNTLoss)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def roi_align(*args):
    op = _get_cache_prim(ROIAlign)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


def ragged_range(*args):
    op = _get_cache_prim(RaggedRange)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def random_categorical(*args):
    op = _get_cache_prim(RandomCategorical)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def random_choice_with_mask(*args):
    op = _get_cache_prim(RandomChoiceWithMask)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def random_gamma(*args):
    op = _get_cache_prim(RandomGamma)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def random_gamma(*args):
    op = _get_cache_prim(RandomGamma)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def random_poisson(*args):
    op = _get_cache_prim(RandomPoisson)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def random_shuffle(*args):
    op = _get_cache_prim(RandomShuffle)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def randperm(*args):
    op = _get_cache_prim(Randperm)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def randperm_v2(*args):
    op = _get_cache_prim(RandpermV2)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def range(*args):
    op = _get_cache_prim(Range)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


rank_op = Rank()
def rank(*args):
    rank_op = hook_call(rank_op)
    return rank_op(*args)


re_lu_op = ReLU()
def re_lu(*args):
    re_lu_op = hook_call(re_lu_op)
    return re_lu_op(*args)


re_lu6_op = ReLU6()
def re_lu6(*args):
    re_lu6_op = hook_call(re_lu6_op)
    return re_lu6_op(*args)


real_op = Real()
def real(*args):
    real_op = hook_call(real_op)
    return real_op(*args)


real_div_op = RealDiv()
def real_div(*args):
    real_div_op = hook_call(real_div_op)
    return real_div_op(*args)


def receive(*args):
    op = _get_cache_prim(Receive)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


reciprocal_op = Reciprocal()
def reciprocal(*args):
    reciprocal_op = hook_call(reciprocal_op)
    return reciprocal_op(*args)


def reduce(*args):
    op = _get_cache_prim(Reduce)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def reduce_all(*args):
    op = _get_cache_prim(ReduceAll)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def reduce_any(*args):
    op = _get_cache_prim(ReduceAny)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def reduce_max(*args):
    op = _get_cache_prim(ReduceMax)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def reduce_mean(*args):
    op = _get_cache_prim(ReduceMean)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def reduce_min(*args):
    op = _get_cache_prim(ReduceMin)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def reduce_prod(*args):
    op = _get_cache_prim(ReduceProd)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def reduce_scatter(*args):
    op = _get_cache_prim(ReduceScatter)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def reduce_scatter_v(*args):
    op = _get_cache_prim(ReduceScatterV)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def reduce_std(*args):
    op = _get_cache_prim(ReduceStd)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def reduce_sum(*args):
    op = _get_cache_prim(ReduceSum)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def renorm(*args):
    op = _get_cache_prim(Renorm)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


reshape_op = Reshape()
def reshape(*args):
    reshape_op = hook_call(reshape_op)
    return reshape_op(*args)


reshape_and_cache_op = ReshapeAndCache()
def reshape_and_cache(*args):
    reshape_and_cache_op = hook_call(reshape_and_cache_op)
    return reshape_and_cache_op(*args)


def reshard(*args):
    op = _get_cache_prim(Reshard)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def resize_area(*args):
    op = _get_cache_prim(ResizeArea)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def resize_bicubic(*args):
    op = _get_cache_prim(ResizeBicubic)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def resize_bilinear_v2(*args):
    op = _get_cache_prim(ResizeBilinearV2)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def resize_linear1_d(*args):
    op = _get_cache_prim(ResizeLinear1D)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def resize_nearest_neighbor(*args):
    op = _get_cache_prim(ResizeNearestNeighbor)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def resize_nearest_neighbor_v2(*args):
    op = _get_cache_prim(ResizeNearestNeighborV2)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


reusing_op = Reusing()
def reusing(*args):
    reusing_op = hook_call(reusing_op)
    return reusing_op(*args)


def reverse_sequence(*args):
    op = _get_cache_prim(ReverseSequence)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def reverse_v2(*args):
    op = _get_cache_prim(ReverseV2)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


right_shift_op = RightShift()
def right_shift(*args):
    right_shift_op = hook_call(right_shift_op)
    return right_shift_op(*args)


rint_op = Rint()
def rint(*args):
    rint_op = hook_call(rint_op)
    return rint_op(*args)


def rms_norm(*args):
    op = _get_cache_prim(RmsNorm)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def roll(*args):
    op = _get_cache_prim(Roll)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


round_op = Round()
def round(*args):
    round_op = hook_call(round_op)
    return round_op(*args)


rsqrt_op = Rsqrt()
def rsqrt(*args):
    rsqrt_op = hook_call(rsqrt_op)
    return rsqrt_op(*args)


def sgd(*args):
    op = _get_cache_prim(SGD)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def stft(*args):
    op = _get_cache_prim(STFT)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def sample_distorted_bounding_box_v2(*args):
    op = _get_cache_prim(SampleDistortedBoundingBoxV2)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


scalar_cast_op = ScalarCast()
def scalar_cast(*args):
    scalar_cast_op = hook_call(scalar_cast_op)
    return scalar_cast_op(*args)


scalar_summary_op = ScalarSummary()
def scalar_summary(*args):
    scalar_summary_op = hook_call(scalar_summary_op)
    return scalar_summary_op(*args)


scalar_to_array_op = ScalarToArray()
def scalar_to_array(*args):
    scalar_to_array_op = hook_call(scalar_to_array_op)
    return scalar_to_array_op(*args)


scalar_to_tensor_op = ScalarToTensor()
def scalar_to_tensor(*args):
    scalar_to_tensor_op = hook_call(scalar_to_tensor_op)
    return scalar_to_tensor_op(*args)


def scale_and_translate(*args):
    op = _get_cache_prim(ScaleAndTranslate)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


scan_op = Scan()
def scan(*args):
    scan_op = hook_call(scan_op)
    return scan_op(*args)


def scatter_add(*args):
    op = _get_cache_prim(ScatterAdd)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_add_with_axis(*args):
    op = _get_cache_prim(ScatterAddWithAxis)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_div(*args):
    op = _get_cache_prim(ScatterDiv)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_max(*args):
    op = _get_cache_prim(ScatterMax)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_min(*args):
    op = _get_cache_prim(ScatterMin)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_mul(*args):
    op = _get_cache_prim(ScatterMul)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


scatter_nd_op = ScatterNd()
def scatter_nd(*args):
    scatter_nd_op = hook_call(scatter_nd_op)
    return scatter_nd_op(*args)


def scatter_nd_add(*args):
    op = _get_cache_prim(ScatterNdAdd)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_nd_div(*args):
    op = _get_cache_prim(ScatterNdDiv)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_nd_max(*args):
    op = _get_cache_prim(ScatterNdMax)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_nd_min(*args):
    op = _get_cache_prim(ScatterNdMin)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_nd_mul(*args):
    op = _get_cache_prim(ScatterNdMul)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_nd_sub(*args):
    op = _get_cache_prim(ScatterNdSub)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_nd_update(*args):
    op = _get_cache_prim(ScatterNdUpdate)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


scatter_non_aliasing_add_op = ScatterNonAliasingAdd()
def scatter_non_aliasing_add(*args):
    scatter_non_aliasing_add_op = hook_call(scatter_non_aliasing_add_op)
    return scatter_non_aliasing_add_op(*args)


def scatter_sub(*args):
    op = _get_cache_prim(ScatterSub)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def scatter_update(*args):
    op = _get_cache_prim(ScatterUpdate)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


se_lu_op = SeLU()
def se_lu(*args):
    se_lu_op = hook_call(se_lu_op)
    return se_lu_op(*args)


def search_sorted(*args):
    op = _get_cache_prim(SearchSorted)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


segment_max_op = SegmentMax()
def segment_max(*args):
    segment_max_op = hook_call(segment_max_op)
    return segment_max_op(*args)


segment_mean_op = SegmentMean()
def segment_mean(*args):
    segment_mean_op = hook_call(segment_mean_op)
    return segment_mean_op(*args)


segment_min_op = SegmentMin()
def segment_min(*args):
    segment_min_op = hook_call(segment_min_op)
    return segment_min_op(*args)


segment_prod_op = SegmentProd()
def segment_prod(*args):
    segment_prod_op = hook_call(segment_prod_op)
    return segment_prod_op(*args)


segment_sum_op = SegmentSum()
def segment_sum(*args):
    segment_sum_op = hook_call(segment_sum_op)
    return segment_sum_op(*args)


select_op = Select()
def select(*args):
    select_op = hook_call(select_op)
    return select_op(*args)


select_view_op = SelectView()
def select_view(*args):
    select_view_op = hook_call(select_view_op)
    return select_view_op(*args)


def send(*args):
    op = _get_cache_prim(Send)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


shape_op = Shape()
def shape(*args):
    shape_op = hook_call(shape_op)
    return shape_op(*args)


sigmoid_op = Sigmoid()
def sigmoid(*args):
    sigmoid_op = hook_call(sigmoid_op)
    return sigmoid_op(*args)


sigmoid_cross_entropy_with_logits_op = SigmoidCrossEntropyWithLogits()
def sigmoid_cross_entropy_with_logits(*args):
    sigmoid_cross_entropy_with_logits_op = hook_call(sigmoid_cross_entropy_with_logits_op)
    return sigmoid_cross_entropy_with_logits_op(*args)


sign_op = Sign()
def sign(*args):
    sign_op = hook_call(sign_op)
    return sign_op(*args)


sin_op = Sin()
def sin(*args):
    sin_op = hook_call(sin_op)
    return sin_op(*args)


sinc_op = Sinc()
def sinc(*args):
    sinc_op = hook_call(sinc_op)
    return sinc_op(*args)


sinh_op = Sinh()
def sinh(*args):
    sinh_op = hook_call(sinh_op)
    return sinh_op(*args)


size_op = Size()
def size(*args):
    size_op = hook_call(size_op)
    return size_op(*args)


slice_op = Slice()
def slice(*args):
    slice_op = hook_call(slice_op)
    return slice_op(*args)


def smooth_l1_loss(*args):
    op = _get_cache_prim(SmoothL1Loss)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def soft_margin_loss(*args):
    op = _get_cache_prim(SoftMarginLoss)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def soft_shrink(*args):
    op = _get_cache_prim(SoftShrink)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def softmax(*args):
    op = _get_cache_prim(Softmax)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


softmax_cross_entropy_with_logits_op = SoftmaxCrossEntropyWithLogits()
def softmax_cross_entropy_with_logits(*args):
    softmax_cross_entropy_with_logits_op = hook_call(softmax_cross_entropy_with_logits_op)
    return softmax_cross_entropy_with_logits_op(*args)


softplus_op = Softplus()
def softplus(*args):
    softplus_op = hook_call(softplus_op)
    return softplus_op(*args)


softsign_op = Softsign()
def softsign(*args):
    softsign_op = hook_call(softsign_op)
    return softsign_op(*args)


def sort(*args):
    op = _get_cache_prim(Sort)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def space_to_batch(*args):
    op = _get_cache_prim(SpaceToBatch)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def space_to_batch_nd(*args):
    op = _get_cache_prim(SpaceToBatchND)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def space_to_depth(*args):
    op = _get_cache_prim(SpaceToDepth)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def sparse_apply_adadelta(*args):
    op = _get_cache_prim(SparseApplyAdadelta)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def sparse_apply_adagrad(*args):
    op = _get_cache_prim(SparseApplyAdagrad)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


def sparse_apply_adagrad_v2(*args):
    op = _get_cache_prim(SparseApplyAdagradV2)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def sparse_apply_ftrl(*args):
    op = _get_cache_prim(SparseApplyFtrl)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


def sparse_apply_ftrl_v2(*args):
    op = _get_cache_prim(SparseApplyFtrlV2)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def sparse_apply_proximal_adagrad(*args):
    op = _get_cache_prim(SparseApplyProximalAdagrad)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def sparse_apply_rms_prop(*args):
    op = _get_cache_prim(SparseApplyRMSProp)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


sparse_gather_v2_op = SparseGatherV2()
def sparse_gather_v2(*args):
    sparse_gather_v2_op = hook_call(sparse_gather_v2_op)
    return sparse_gather_v2_op(*args)


sparse_slice_op = SparseSlice()
def sparse_slice(*args):
    sparse_slice_op = hook_call(sparse_slice_op)
    return sparse_slice_op(*args)


def sparse_softmax_cross_entropy_with_logits(*args):
    op = _get_cache_prim(SparseSoftmaxCrossEntropyWithLogits)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


sparse_tensor_dense_add_op = SparseTensorDenseAdd()
def sparse_tensor_dense_add(*args):
    sparse_tensor_dense_add_op = hook_call(sparse_tensor_dense_add_op)
    return sparse_tensor_dense_add_op(*args)


def sparse_tensor_dense_matmul(*args):
    op = _get_cache_prim(SparseTensorDenseMatmul)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


sparse_to_dense_op = SparseToDense()
def sparse_to_dense(*args):
    sparse_to_dense_op = hook_call(sparse_to_dense_op)
    return sparse_to_dense_op(*args)


def split(*args):
    op = _get_cache_prim(Split)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def split_v(*args):
    op = _get_cache_prim(SplitV)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


sqrt_op = Sqrt()
def sqrt(*args):
    sqrt_op = hook_call(sqrt_op)
    return sqrt_op(*args)


square_op = Square()
def square(*args):
    square_op = hook_call(square_op)
    return square_op(*args)


square_sum_all_op = SquareSumAll()
def square_sum_all(*args):
    square_sum_all_op = hook_call(square_sum_all_op)
    return square_sum_all_op(*args)


squared_difference_op = SquaredDifference()
def squared_difference(*args):
    squared_difference_op = hook_call(squared_difference_op)
    return squared_difference_op(*args)


def squeeze(*args):
    op = _get_cache_prim(Squeeze)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def stack(*args):
    op = _get_cache_prim(Stack)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def standard_laplace(*args):
    op = _get_cache_prim(StandardLaplace)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def standard_normal(*args):
    op = _get_cache_prim(StandardNormal)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


stop_gradient_op = StopGradient()
def stop_gradient(*args):
    stop_gradient_op = hook_call(stop_gradient_op)
    return stop_gradient_op(*args)


def strided_slice(*args):
    op = _get_cache_prim(StridedSlice)(*args[-5:])
    op = hook_call(op)
    return op(*args[:-5])


sub_op = Sub()
def sub(*args):
    sub_op = hook_call(sub_op)
    return sub_op(*args)


sub_and_filter_op = SubAndFilter()
def sub_and_filter(*args):
    sub_and_filter_op = hook_call(sub_and_filter_op)
    return sub_and_filter_op(*args)


def svd(*args):
    op = _get_cache_prim(Svd)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


tan_op = Tan()
def tan(*args):
    tan_op = hook_call(tan_op)
    return tan_op(*args)


tanh_op = Tanh()
def tanh(*args):
    tanh_op = hook_call(tanh_op)
    return tanh_op(*args)


tensor_add_op = TensorAdd()
def tensor_add(*args):
    tensor_add_op = hook_call(tensor_add_op)
    return tensor_add_op(*args)


def tensor_dump(*args):
    op = _get_cache_prim(TensorDump)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


tensor_scatter_add_op = TensorScatterAdd()
def tensor_scatter_add(*args):
    tensor_scatter_add_op = hook_call(tensor_scatter_add_op)
    return tensor_scatter_add_op(*args)


tensor_scatter_div_op = TensorScatterDiv()
def tensor_scatter_div(*args):
    tensor_scatter_div_op = hook_call(tensor_scatter_div_op)
    return tensor_scatter_div_op(*args)


def tensor_scatter_elements(*args):
    op = _get_cache_prim(TensorScatterElements)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


tensor_scatter_max_op = TensorScatterMax()
def tensor_scatter_max(*args):
    tensor_scatter_max_op = hook_call(tensor_scatter_max_op)
    return tensor_scatter_max_op(*args)


tensor_scatter_min_op = TensorScatterMin()
def tensor_scatter_min(*args):
    tensor_scatter_min_op = hook_call(tensor_scatter_min_op)
    return tensor_scatter_min_op(*args)


tensor_scatter_mul_op = TensorScatterMul()
def tensor_scatter_mul(*args):
    tensor_scatter_mul_op = hook_call(tensor_scatter_mul_op)
    return tensor_scatter_mul_op(*args)


tensor_scatter_sub_op = TensorScatterSub()
def tensor_scatter_sub(*args):
    tensor_scatter_sub_op = hook_call(tensor_scatter_sub_op)
    return tensor_scatter_sub_op(*args)


tensor_scatter_update_op = TensorScatterUpdate()
def tensor_scatter_update(*args):
    tensor_scatter_update_op = hook_call(tensor_scatter_update_op)
    return tensor_scatter_update_op(*args)


tensor_shape_op = TensorShape()
def tensor_shape(*args):
    tensor_shape_op = hook_call(tensor_shape_op)
    return tensor_shape_op(*args)


tensor_summary_op = TensorSummary()
def tensor_summary(*args):
    tensor_summary_op = hook_call(tensor_summary_op)
    return tensor_summary_op(*args)


tile_op = Tile()
def tile(*args):
    tile_op = hook_call(tile_op)
    return tile_op(*args)


def top_k(*args):
    op = _get_cache_prim(TopK)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


trace_op = Trace()
def trace(*args):
    trace_op = hook_call(trace_op)
    return trace_op(*args)


transpose_op = Transpose()
def transpose(*args):
    transpose_op = hook_call(transpose_op)
    return transpose_op(*args)


transpose_ext_view_op = TransposeExtView()
def transpose_ext_view(*args):
    transpose_ext_view_op = hook_call(transpose_ext_view_op)
    return transpose_ext_view_op(*args)


transpose_view_op = TransposeView()
def transpose_view(*args):
    transpose_view_op = hook_call(transpose_view_op)
    return transpose_view_op(*args)


tridiagonal_mat_mul_op = TridiagonalMatMul()
def tridiagonal_mat_mul(*args):
    tridiagonal_mat_mul_op = hook_call(tridiagonal_mat_mul_op)
    return tridiagonal_mat_mul_op(*args)


def tril(*args):
    op = _get_cache_prim(Tril)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def tril_indices(*args):
    op = _get_cache_prim(TrilIndices)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def triplet_margin_loss(*args):
    op = _get_cache_prim(TripletMarginLoss)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


def triu(*args):
    op = _get_cache_prim(Triu)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


def triu_indices(*args):
    op = _get_cache_prim(TriuIndices)(*args[-4:])
    op = hook_call(op)
    return op(*args[:-4])


trunc_op = Trunc()
def trunc(*args):
    trunc_op = hook_call(trunc_op)
    return trunc_op(*args)


truncate_div_op = TruncateDiv()
def truncate_div(*args):
    truncate_div_op = hook_call(truncate_div_op)
    return truncate_div_op(*args)


truncate_mod_op = TruncateMod()
def truncate_mod(*args):
    truncate_mod_op = hook_call(truncate_mod_op)
    return truncate_mod_op(*args)


def truncated_normal(*args):
    op = _get_cache_prim(TruncatedNormal)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


tuple_to_array_op = TupleToArray()
def tuple_to_array(*args):
    tuple_to_array_op = hook_call(tuple_to_array_op)
    return tuple_to_array_op(*args)


def uniform_candidate_sampler(*args):
    op = _get_cache_prim(UniformCandidateSampler)(*args[-6:])
    op = hook_call(op)
    return op(*args[:-6])


def uniform_int(*args):
    op = _get_cache_prim(UniformInt)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


def uniform_real(*args):
    op = _get_cache_prim(UniformReal)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


unique_op = Unique()
def unique(*args):
    unique_op = hook_call(unique_op)
    return unique_op(*args)


def unique_consecutive(*args):
    op = _get_cache_prim(UniqueConsecutive)(*args[-3:])
    op = hook_call(op)
    return op(*args[:-3])


unique_with_pad_op = UniqueWithPad()
def unique_with_pad(*args):
    unique_with_pad_op = hook_call(unique_with_pad_op)
    return unique_with_pad_op(*args)


def unpack(*args):
    op = _get_cache_prim(Unpack)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


unravel_index_op = UnravelIndex()
def unravel_index(*args):
    unravel_index_op = hook_call(unravel_index_op)
    return unravel_index_op(*args)


unsorted_segment_max_op = UnsortedSegmentMax()
def unsorted_segment_max(*args):
    unsorted_segment_max_op = hook_call(unsorted_segment_max_op)
    return unsorted_segment_max_op(*args)


unsorted_segment_min_op = UnsortedSegmentMin()
def unsorted_segment_min(*args):
    unsorted_segment_min_op = hook_call(unsorted_segment_min_op)
    return unsorted_segment_min_op(*args)


unsorted_segment_prod_op = UnsortedSegmentProd()
def unsorted_segment_prod(*args):
    unsorted_segment_prod_op = hook_call(unsorted_segment_prod_op)
    return unsorted_segment_prod_op(*args)


unsorted_segment_sum_op = UnsortedSegmentSum()
def unsorted_segment_sum(*args):
    unsorted_segment_sum_op = hook_call(unsorted_segment_sum_op)
    return unsorted_segment_sum_op(*args)


def unstack(*args):
    op = _get_cache_prim(Unstack)(*args[-2:])
    op = hook_call(op)
    return op(*args[:-2])


update_state_op = UpdateState()
def update_state(*args):
    update_state_op = hook_call(update_state_op)
    return update_state_op(*args)


def upper_bound(*args):
    op = _get_cache_prim(UpperBound)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


upsample_nearest3_d_op = UpsampleNearest3D()
def upsample_nearest3_d(*args):
    upsample_nearest3_d_op = hook_call(upsample_nearest3_d_op)
    return upsample_nearest3_d_op(*args)


def upsample_trilinear3_d(*args):
    op = _get_cache_prim(UpsampleTrilinear3D)(*args[-1:])
    op = hook_call(op)
    return op(*args[:-1])


while_loop_op = WhileLoop()
def while_loop(*args):
    while_loop_op = hook_call(while_loop_op)
    return while_loop_op(*args)


xdivy_op = Xdivy()
def xdivy(*args):
    xdivy_op = hook_call(xdivy_op)
    return xdivy_op(*args)


xlogy_op = Xlogy()
def xlogy(*args):
    xlogy_op = hook_call(xlogy_op)
    return xlogy_op(*args)


zeros_op = Zeros()
def zeros(*args):
    zeros_op = hook_call(zeros_op)
    return zeros_op(*args)


zeros_like_op = ZerosLike()
def zeros_like(*args):
    zeros_like_op = hook_call(zeros_like_op)
    return zeros_like_op(*args)


zeta_op = Zeta()
def zeta(*args):
    zeta_op = hook_call(zeta_op)
    return zeta_op(*args)

