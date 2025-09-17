from mindspore.ops.operations import *
from mindspore.ops.operations._grad_ops import *
from mindspore.ops.operations._inner_ops import *
from mindspore.ops._primitive_cache import _get_cache_prim


a_cos_grad_op = ACosGrad().set_device('GPU')
def a_cos_grad(*args):
    return a_cos_grad_op(*args)


abs_grad_op = AbsGrad().set_device('GPU')
def abs_grad(*args):
    return abs_grad_op(*args)


acosh_grad_op = AcoshGrad().set_device('GPU')
def acosh_grad(*args):
    return acosh_grad_op(*args)


adaptive_avg_pool2_d_grad_op = AdaptiveAvgPool2DGrad().set_device('GPU')
def adaptive_avg_pool2_d_grad(*args):
    return adaptive_avg_pool2_d_grad_op(*args)


adaptive_avg_pool3_d_grad_op = AdaptiveAvgPool3DGrad().set_device('GPU')
def adaptive_avg_pool3_d_grad(*args):
    return adaptive_avg_pool3_d_grad_op(*args)


adaptive_max_pool2_d_grad_op = AdaptiveMaxPool2DGrad().set_device('GPU')
def adaptive_max_pool2_d_grad(*args):
    return adaptive_max_pool2_d_grad_op(*args)


adaptive_max_pool3_d_grad_op = AdaptiveMaxPool3DGrad().set_device('GPU')
def adaptive_max_pool3_d_grad(*args):
    return adaptive_max_pool3_d_grad_op(*args)


def affine_grid_grad(*args):
    op = _get_cache_prim(AffineGridGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


asin_grad_op = AsinGrad().set_device('GPU')
def asin_grad(*args):
    return asin_grad_op(*args)


asinh_grad_op = AsinhGrad().set_device('GPU')
def asinh_grad(*args):
    return asinh_grad_op(*args)


atan_grad_op = AtanGrad().set_device('GPU')
def atan_grad(*args):
    return atan_grad_op(*args)


def avg_pool3_d_grad(*args):
    op = _get_cache_prim(AvgPool3DGrad)(*args[-8:]).set_device('GPU')
    return op(*args[:-8])


def avg_pool_grad(*args):
    op = _get_cache_prim(AvgPoolGrad)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def avg_pool_grad_ge(*args):
    op = _get_cache_prim(AvgPoolGradGe)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def avg_pool_grad_v1(*args):
    op = _get_cache_prim(AvgPoolGradV1)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def avg_pool_grad_vm(*args):
    op = _get_cache_prim(AvgPoolGradVm)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def bn_training_reduce_grad(*args):
    op = _get_cache_prim(BNTrainingReduceGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def bn_training_update_grad(*args):
    op = _get_cache_prim(BNTrainingUpdateGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def basic_lstm_cell_c_state_grad(*args):
    op = _get_cache_prim(BasicLSTMCellCStateGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def basic_lstm_cell_input_grad(*args):
    op = _get_cache_prim(BasicLSTMCellInputGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


basic_lstm_cell_weight_grad_op = BasicLSTMCellWeightGrad().set_device('GPU')
def basic_lstm_cell_weight_grad(*args):
    return basic_lstm_cell_weight_grad_op(*args)


def batch_norm_grad(*args):
    op = _get_cache_prim(BatchNormGrad)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def batch_norm_grad_grad(*args):
    op = _get_cache_prim(BatchNormGradGrad)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def bias_add_grad(*args):
    op = _get_cache_prim(BiasAddGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def binary_cross_entropy_grad(*args):
    op = _get_cache_prim(BinaryCrossEntropyGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


cholesky_grad_op = CholeskyGrad().set_device('GPU')
def cholesky_grad(*args):
    return cholesky_grad_op(*args)


def concat_offset(*args):
    op = _get_cache_prim(ConcatOffset)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def conv2_d_backprop_filter(*args):
    op = _get_cache_prim(Conv2DBackpropFilter)(*args[-10:]).set_device('GPU')
    return op(*args[:-10])


def conv3_d_backprop_filter(*args):
    op = _get_cache_prim(Conv3DBackpropFilter)(*args[-9:]).set_device('GPU')
    return op(*args[:-9])


def deformable_offsets_grad(*args):
    op = _get_cache_prim(DeformableOffsetsGrad)(*args[-7:]).set_device('GPU')
    return op(*args[:-7])


def depthwise_conv2d_native_backprop_filter(*args):
    op = _get_cache_prim(DepthwiseConv2dNativeBackpropFilter)(*args[-9:]).set_device('GPU')
    return op(*args[:-9])


def depthwise_conv2d_native_backprop_input(*args):
    op = _get_cache_prim(DepthwiseConv2dNativeBackpropInput)(*args[-9:]).set_device('GPU')
    return op(*args[:-9])


def dilation2_d_backprop_filter(*args):
    op = _get_cache_prim(Dilation2DBackpropFilter)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def dilation2_d_backprop_input(*args):
    op = _get_cache_prim(Dilation2DBackpropInput)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def dropout_grad(*args):
    op = _get_cache_prim(DropoutGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def dynamic_gruv2_grad(*args):
    op = _get_cache_prim(DynamicGRUV2Grad)(*args[-8:]).set_device('GPU')
    return op(*args[:-8])


def dynamic_rnn_grad(*args):
    op = _get_cache_prim(DynamicRNNGrad)(*args[-9:]).set_device('GPU')
    return op(*args[:-9])


def einsum_grad(*args):
    op = _get_cache_prim(EinsumGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


elu_grad_op = EluGrad().set_device('GPU')
def elu_grad(*args):
    return elu_grad_op(*args)


embedding_lookup_comm_grad_op = EmbeddingLookupCommGrad().set_device('GPU')
def embedding_lookup_comm_grad(*args):
    return embedding_lookup_comm_grad_op(*args)


fast_ge_lu_grad_op = FastGeLUGrad().set_device('GPU')
def fast_ge_lu_grad(*args):
    return fast_ge_lu_grad_op(*args)


def flash_attention_score_grad(*args):
    op = _get_cache_prim(FlashAttentionScoreGrad)(*args[-8:]).set_device('GPU')
    return op(*args[:-8])


flatten_grad_op = FlattenGrad().set_device('GPU')
def flatten_grad(*args):
    return flatten_grad_op(*args)


def fractional_avg_pool_grad(*args):
    op = _get_cache_prim(FractionalAvgPoolGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def fractional_max_pool3_d_grad_with_fixed_ksize(*args):
    op = _get_cache_prim(FractionalMaxPool3DGradWithFixedKsize)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def fractional_max_pool_grad(*args):
    op = _get_cache_prim(FractionalMaxPoolGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def fractional_max_pool_grad_with_fixed_ksize(*args):
    op = _get_cache_prim(FractionalMaxPoolGradWithFixedKsize)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def gruv2_grad(*args):
    op = _get_cache_prim(GRUV2Grad)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


gather_d_grad_v2_op = GatherDGradV2().set_device('GPU')
def gather_d_grad_v2(*args):
    return gather_d_grad_v2_op(*args)


ge_lu_grad_op = GeLUGrad().set_device('GPU')
def ge_lu_grad(*args):
    return ge_lu_grad_op(*args)


def global_comm(*args):
    op = _get_cache_prim(GlobalComm)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def glu_grad(*args):
    op = _get_cache_prim(GluGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def grid_sampler2_d_grad(*args):
    op = _get_cache_prim(GridSampler2DGrad)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def grid_sampler3_d_grad(*args):
    op = _get_cache_prim(GridSampler3DGrad)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def gru_grad_data(*args):
    op = _get_cache_prim(GruGradData)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def gru_grad_weight(*args):
    op = _get_cache_prim(GruGradWeight)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def h_shrink_grad(*args):
    op = _get_cache_prim(HShrinkGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


h_sigmoid_grad_op = HSigmoidGrad().set_device('GPU')
def h_sigmoid_grad(*args):
    return h_sigmoid_grad_op(*args)


h_swish_grad_op = HSwishGrad().set_device('GPU')
def h_swish_grad(*args):
    return h_swish_grad_op(*args)


igamma_grad_a_op = IgammaGradA().set_device('GPU')
def igamma_grad_a(*args):
    return igamma_grad_a_op(*args)


def instance_norm_grad(*args):
    op = _get_cache_prim(InstanceNormGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def instance_norm_v2_grad(*args):
    op = _get_cache_prim(InstanceNormV2Grad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


inv_grad_op = InvGrad().set_device('GPU')
def inv_grad(*args):
    return inv_grad_op(*args)


def kl_div_loss_grad(*args):
    op = _get_cache_prim(KLDivLossGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def l2_normalize_grad(*args):
    op = _get_cache_prim(L2NormalizeGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def lrn_grad(*args):
    op = _get_cache_prim(LRNGrad)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def lstm_grad(*args):
    op = _get_cache_prim(LSTMGrad)(*args[-7:]).set_device('GPU')
    return op(*args[:-7])


def lstm_grad_data(*args):
    op = _get_cache_prim(LSTMGradData)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def lstm_grad_weight(*args):
    op = _get_cache_prim(LSTMGradWeight)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def layer_norm_grad(*args):
    op = _get_cache_prim(LayerNormGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def layer_norm_grad_grad(*args):
    op = _get_cache_prim(LayerNormGradGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def log_softmax_grad(*args):
    op = _get_cache_prim(LogSoftmaxGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def logit_grad(*args):
    op = _get_cache_prim(LogitGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def lu_unpack_grad(*args):
    op = _get_cache_prim(LuUnpackGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


map_tensor_get_grad_op = MapTensorGetGrad().set_device('GPU')
def map_tensor_get_grad(*args):
    return map_tensor_get_grad_op(*args)


masked_select_grad_op = MaskedSelectGrad().set_device('GPU')
def masked_select_grad(*args):
    return masked_select_grad_op(*args)


def max_pool3_d_grad(*args):
    op = _get_cache_prim(MaxPool3DGrad)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


def max_pool3_d_grad_grad(*args):
    op = _get_cache_prim(MaxPool3DGradGrad)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def max_pool3_d_grad_with_argmax(*args):
    op = _get_cache_prim(MaxPool3DGradWithArgmax)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def max_pool_grad(*args):
    op = _get_cache_prim(MaxPoolGrad)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def max_pool_grad_grad(*args):
    op = _get_cache_prim(MaxPoolGradGrad)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def max_pool_grad_grad_with_argmax(*args):
    op = _get_cache_prim(MaxPoolGradGradWithArgmax)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def max_pool_grad_v1(*args):
    op = _get_cache_prim(MaxPoolGradV1)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def max_pool_grad_with_argmax(*args):
    op = _get_cache_prim(MaxPoolGradWithArgmax)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def max_pool_grad_with_argmax_v2(*args):
    op = _get_cache_prim(MaxPoolGradWithArgmaxV2)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def max_unpool2_d_grad(*args):
    op = _get_cache_prim(MaxUnpool2DGrad)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


def max_unpool3_d_grad(*args):
    op = _get_cache_prim(MaxUnpool3DGrad)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


def maximum_grad(*args):
    op = _get_cache_prim(MaximumGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def maximum_grad_grad(*args):
    op = _get_cache_prim(MaximumGradGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def median_grad(*args):
    op = _get_cache_prim(MedianGrad)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def minimum_grad(*args):
    op = _get_cache_prim(MinimumGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


minimum_grad_grad_op = MinimumGradGrad().set_device('GPU')
def minimum_grad_grad(*args):
    return minimum_grad_grad_op(*args)


def mirror_pad_grad(*args):
    op = _get_cache_prim(MirrorPadGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def multi_margin_loss_grad(*args):
    op = _get_cache_prim(MultiMarginLossGrad)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def multilabel_margin_loss_grad(*args):
    op = _get_cache_prim(MultilabelMarginLossGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def mvlgamma_grad(*args):
    op = _get_cache_prim(MvlgammaGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def nll_loss_grad(*args):
    op = _get_cache_prim(NLLLossGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def neighbor_exchange_v2_grad(*args):
    op = _get_cache_prim(NeighborExchangeV2Grad)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


p_re_lu_grad_op = PReLUGrad().set_device('GPU')
def p_re_lu_grad(*args):
    return p_re_lu_grad_op(*args)


def psroi_pooling_grad(*args):
    op = _get_cache_prim(PSROIPoolingGrad)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def pad_v3_grad(*args):
    op = _get_cache_prim(PadV3Grad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def parallel_resize_bilinear_grad(*args):
    op = _get_cache_prim(ParallelResizeBilinearGrad)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def pdist_grad(*args):
    op = _get_cache_prim(PdistGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def primitive(*args):
    op = _get_cache_prim(Primitive)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def primitive_with_infer(*args):
    op = _get_cache_prim(PrimitiveWithInfer)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def ps_roi_pooling_grad(*args):
    op = _get_cache_prim(PsROIPoolingGrad)(*args[-9:]).set_device('GPU')
    return op(*args[:-9])


def roi_align_grad(*args):
    op = _get_cache_prim(ROIAlignGrad)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


random_gamma_grad_op = RandomGammaGrad().set_device('GPU')
def random_gamma_grad(*args):
    return random_gamma_grad_op(*args)


re_lu6_grad_op = ReLU6Grad().set_device('GPU')
def re_lu6_grad(*args):
    return re_lu6_grad_op(*args)


reciprocal_grad_op = ReciprocalGrad().set_device('GPU')
def reciprocal_grad(*args):
    return reciprocal_grad_op(*args)


ref_to_embed_op = RefToEmbed().set_device('GPU')
def ref_to_embed(*args):
    return ref_to_embed_op(*args)


relu_grad_op = ReluGrad().set_device('GPU')
def relu_grad(*args):
    return relu_grad_op(*args)


def resize_bicubic_grad(*args):
    op = _get_cache_prim(ResizeBicubicGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def resize_bilinear_grad(*args):
    op = _get_cache_prim(ResizeBilinearGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def resize_linear1_d_grad(*args):
    op = _get_cache_prim(ResizeLinear1DGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def resize_nearest_neighbor_grad(*args):
    op = _get_cache_prim(ResizeNearestNeighborGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def resize_nearest_neighbor_v2_grad(*args):
    op = _get_cache_prim(ResizeNearestNeighborV2Grad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def resize_v2_grad(*args):
    op = _get_cache_prim(ResizeV2Grad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


rms_norm_grad_op = RmsNormGrad().set_device('GPU')
def rms_norm_grad(*args):
    return rms_norm_grad_op(*args)


rsqrt_grad_op = RsqrtGrad().set_device('GPU')
def rsqrt_grad(*args):
    return rsqrt_grad_op(*args)


def scale_and_translate_grad(*args):
    op = _get_cache_prim(ScaleAndTranslateGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


selu_grad_op = SeluGrad().set_device('GPU')
def selu_grad(*args):
    return selu_grad_op(*args)


si_lu_grad_op = SiLUGrad().set_device('GPU')
def si_lu_grad(*args):
    return si_lu_grad_op(*args)


sigmoid_cross_entropy_with_logits_grad_op = SigmoidCrossEntropyWithLogitsGrad().set_device('GPU')
def sigmoid_cross_entropy_with_logits_grad(*args):
    return sigmoid_cross_entropy_with_logits_grad_op(*args)


sigmoid_grad_op = SigmoidGrad().set_device('GPU')
def sigmoid_grad(*args):
    return sigmoid_grad_op(*args)


slice_grad_op = SliceGrad().set_device('GPU')
def slice_grad(*args):
    return slice_grad_op(*args)


def smooth_l1_loss_grad(*args):
    op = _get_cache_prim(SmoothL1LossGrad)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def soft_margin_loss_grad(*args):
    op = _get_cache_prim(SoftMarginLossGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def soft_shrink_grad(*args):
    op = _get_cache_prim(SoftShrinkGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


softmax_grad_op = SoftmaxGrad().set_device('GPU')
def softmax_grad(*args):
    return softmax_grad_op(*args)


softplus_grad_op = SoftplusGrad().set_device('GPU')
def softplus_grad(*args):
    return softplus_grad_op(*args)


sparse_fill_empty_rows_grad_op = SparseFillEmptyRowsGrad().set_device('GPU')
def sparse_fill_empty_rows_grad(*args):
    return sparse_fill_empty_rows_grad_op(*args)


sparse_segment_mean_grad_op = SparseSegmentMeanGrad().set_device('GPU')
def sparse_segment_mean_grad(*args):
    return sparse_segment_mean_grad_op(*args)


sparse_segment_sqrt_n_grad_op = SparseSegmentSqrtNGrad().set_device('GPU')
def sparse_segment_sqrt_n_grad(*args):
    return sparse_segment_sqrt_n_grad_op(*args)


sparse_segment_sum_grad_op = SparseSegmentSumGrad().set_device('GPU')
def sparse_segment_sum_grad(*args):
    return sparse_segment_sum_grad_op(*args)


sparse_slice_grad_op = SparseSliceGrad().set_device('GPU')
def sparse_slice_grad(*args):
    return sparse_slice_grad_op(*args)


sqrt_grad_op = SqrtGrad().set_device('GPU')
def sqrt_grad(*args):
    return sqrt_grad_op(*args)


def strided_slice_grad(*args):
    op = _get_cache_prim(StridedSliceGrad)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


def sync_batch_norm_grad(*args):
    op = _get_cache_prim(SyncBatchNormGrad)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


tanh_grad_op = TanhGrad().set_device('GPU')
def tanh_grad(*args):
    return tanh_grad_op(*args)


trace_grad_op = TraceGrad().set_device('GPU')
def trace_grad(*args):
    return trace_grad_op(*args)


unique_grad_op = UniqueGrad().set_device('GPU')
def unique_grad(*args):
    return unique_grad_op(*args)


upsample_nearest3_d_grad_op = UpsampleNearest3DGrad().set_device('GPU')
def upsample_nearest3_d_grad(*args):
    return upsample_nearest3_d_grad_op(*args)


def upsample_trilinear3_d_grad(*args):
    op = _get_cache_prim(UpsampleTrilinear3DGrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


wkv_grad_op = WKVGrad().set_device('GPU')
def wkv_grad(*args):
    return wkv_grad_op(*args)


a_cos_op = ACos().set_device('GPU')
def a_cos(*args):
    return a_cos_op(*args)


abs_op = Abs().set_device('GPU')
def abs(*args):
    return abs_op(*args)


accumulate_nv2_op = AccumulateNV2().set_device('GPU')
def accumulate_nv2(*args):
    return accumulate_nv2_op(*args)


acosh_op = Acosh().set_device('GPU')
def acosh(*args):
    return acosh_op(*args)


def adam(*args):
    op = _get_cache_prim(Adam)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def adam_no_update_param(*args):
    op = _get_cache_prim(AdamNoUpdateParam)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def adam_weight_decay(*args):
    op = _get_cache_prim(AdamWeightDecay)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def adaptive_avg_pool2_d(*args):
    op = _get_cache_prim(AdaptiveAvgPool2D)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def adaptive_avg_pool3_d(*args):
    op = _get_cache_prim(AdaptiveAvgPool3D)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def adaptive_max_pool2_d(*args):
    op = _get_cache_prim(AdaptiveMaxPool2D)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


adaptive_max_pool3_d_op = AdaptiveMaxPool3D().set_device('GPU')
def adaptive_max_pool3_d(*args):
    return adaptive_max_pool3_d_op(*args)


add_op = Add().set_device('GPU')
def add(*args):
    return add_op(*args)


add_n_op = AddN().set_device('GPU')
def add_n(*args):
    return add_n_op(*args)


addcdiv_op = Addcdiv().set_device('GPU')
def addcdiv(*args):
    return addcdiv_op(*args)


addcmul_op = Addcmul().set_device('GPU')
def addcmul(*args):
    return addcmul_op(*args)


adjust_hue_op = AdjustHue().set_device('GPU')
def adjust_hue(*args):
    return adjust_hue_op(*args)


adjust_saturation_op = AdjustSaturation().set_device('GPU')
def adjust_saturation(*args):
    return adjust_saturation_op(*args)


def affine_grid(*args):
    op = _get_cache_prim(AffineGrid)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def all_gather(*args):
    op = _get_cache_prim(AllGather)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def all_reduce(*args):
    op = _get_cache_prim(AllReduce)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def allto_all(*args):
    op = _get_cache_prim(AlltoAll)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def allto_all_v(*args):
    op = _get_cache_prim(AlltoAllV)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


angle_op = Angle().set_device('GPU')
def angle(*args):
    return angle_op(*args)


apply_ada_max_op = ApplyAdaMax().set_device('GPU')
def apply_ada_max(*args):
    return apply_ada_max_op(*args)


apply_adadelta_op = ApplyAdadelta().set_device('GPU')
def apply_adadelta(*args):
    return apply_adadelta_op(*args)


def apply_adagrad(*args):
    op = _get_cache_prim(ApplyAdagrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def apply_adagrad_da(*args):
    op = _get_cache_prim(ApplyAdagradDA)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def apply_adagrad_v2(*args):
    op = _get_cache_prim(ApplyAdagradV2)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def apply_adam_with_amsgrad(*args):
    op = _get_cache_prim(ApplyAdamWithAmsgrad)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def apply_adam_with_amsgrad_v2(*args):
    op = _get_cache_prim(ApplyAdamWithAmsgradV2)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


apply_add_sign_op = ApplyAddSign().set_device('GPU')
def apply_add_sign(*args):
    return apply_add_sign_op(*args)


def apply_centered_rms_prop(*args):
    op = _get_cache_prim(ApplyCenteredRMSProp)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def apply_ftrl(*args):
    op = _get_cache_prim(ApplyFtrl)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


apply_gradient_descent_op = ApplyGradientDescent().set_device('GPU')
def apply_gradient_descent(*args):
    return apply_gradient_descent_op(*args)


def apply_keras_momentum(*args):
    op = _get_cache_prim(ApplyKerasMomentum)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def apply_momentum(*args):
    op = _get_cache_prim(ApplyMomentum)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


apply_power_sign_op = ApplyPowerSign().set_device('GPU')
def apply_power_sign(*args):
    return apply_power_sign_op(*args)


def apply_proximal_adagrad(*args):
    op = _get_cache_prim(ApplyProximalAdagrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


apply_proximal_gradient_descent_op = ApplyProximalGradientDescent().set_device('GPU')
def apply_proximal_gradient_descent(*args):
    return apply_proximal_gradient_descent_op(*args)


def apply_rms_prop(*args):
    op = _get_cache_prim(ApplyRMSProp)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def apply_rotary_pos_emb(*args):
    op = _get_cache_prim(ApplyRotaryPosEmb)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def approximate_equal(*args):
    op = _get_cache_prim(ApproximateEqual)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def arg_max_with_value(*args):
    op = _get_cache_prim(ArgMaxWithValue)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def arg_min_with_value(*args):
    op = _get_cache_prim(ArgMinWithValue)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def argmax(*args):
    op = _get_cache_prim(Argmax)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def argmin(*args):
    op = _get_cache_prim(Argmin)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


asin_op = Asin().set_device('GPU')
def asin(*args):
    return asin_op(*args)


asinh_op = Asinh().set_device('GPU')
def asinh(*args):
    return asinh_op(*args)


assign_op = Assign().set_device('GPU')
def assign(*args):
    return assign_op(*args)


assign_add_op = AssignAdd().set_device('GPU')
def assign_add(*args):
    return assign_add_op(*args)


assign_sub_op = AssignSub().set_device('GPU')
def assign_sub(*args):
    return assign_sub_op(*args)


atan_op = Atan().set_device('GPU')
def atan(*args):
    return atan_op(*args)


atan2_op = Atan2().set_device('GPU')
def atan2(*args):
    return atan2_op(*args)


atanh_op = Atanh().set_device('GPU')
def atanh(*args):
    return atanh_op(*args)


def avg_pool(*args):
    op = _get_cache_prim(AvgPool)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def avg_pool3_d(*args):
    op = _get_cache_prim(AvgPool3D)(*args[-8:]).set_device('GPU')
    return op(*args[:-8])


def bce_with_logits_loss(*args):
    op = _get_cache_prim(BCEWithLogitsLoss)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def barrier(*args):
    op = _get_cache_prim(Barrier)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def bartlett_window(*args):
    op = _get_cache_prim(BartlettWindow)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def basic_lstm_cell(*args):
    op = _get_cache_prim(BasicLSTMCell)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def batch_i_send_i_recv(*args):
    op = _get_cache_prim(BatchISendIRecv)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


def batch_mat_mul(*args):
    op = _get_cache_prim(BatchMatMul)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def batch_norm(*args):
    op = _get_cache_prim(BatchNorm)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def batch_to_space(*args):
    op = _get_cache_prim(BatchToSpace)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def batch_to_space_nd(*args):
    op = _get_cache_prim(BatchToSpaceND)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


batch_to_space_ndv2_op = BatchToSpaceNDV2().set_device('GPU')
def batch_to_space_ndv2(*args):
    return batch_to_space_ndv2_op(*args)


def bernoulli(*args):
    op = _get_cache_prim(Bernoulli)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


bessel_i0_op = BesselI0().set_device('GPU')
def bessel_i0(*args):
    return bessel_i0_op(*args)


bessel_i0e_op = BesselI0e().set_device('GPU')
def bessel_i0e(*args):
    return bessel_i0e_op(*args)


bessel_i1_op = BesselI1().set_device('GPU')
def bessel_i1(*args):
    return bessel_i1_op(*args)


bessel_i1e_op = BesselI1e().set_device('GPU')
def bessel_i1e(*args):
    return bessel_i1e_op(*args)


bessel_j0_op = BesselJ0().set_device('GPU')
def bessel_j0(*args):
    return bessel_j0_op(*args)


bessel_j1_op = BesselJ1().set_device('GPU')
def bessel_j1(*args):
    return bessel_j1_op(*args)


bessel_k0_op = BesselK0().set_device('GPU')
def bessel_k0(*args):
    return bessel_k0_op(*args)


bessel_k0e_op = BesselK0e().set_device('GPU')
def bessel_k0e(*args):
    return bessel_k0e_op(*args)


bessel_k1_op = BesselK1().set_device('GPU')
def bessel_k1(*args):
    return bessel_k1_op(*args)


bessel_k1e_op = BesselK1e().set_device('GPU')
def bessel_k1e(*args):
    return bessel_k1e_op(*args)


bessel_y0_op = BesselY0().set_device('GPU')
def bessel_y0(*args):
    return bessel_y0_op(*args)


bessel_y1_op = BesselY1().set_device('GPU')
def bessel_y1(*args):
    return bessel_y1_op(*args)


betainc_op = Betainc().set_device('GPU')
def betainc(*args):
    return betainc_op(*args)


def bias_add(*args):
    op = _get_cache_prim(BiasAdd)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def binary_cross_entropy(*args):
    op = _get_cache_prim(BinaryCrossEntropy)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


bincount_op = Bincount().set_device('GPU')
def bincount(*args):
    return bincount_op(*args)


bitwise_and_op = BitwiseAnd().set_device('GPU')
def bitwise_and(*args):
    return bitwise_and_op(*args)


bitwise_or_op = BitwiseOr().set_device('GPU')
def bitwise_or(*args):
    return bitwise_or_op(*args)


bitwise_xor_op = BitwiseXor().set_device('GPU')
def bitwise_xor(*args):
    return bitwise_xor_op(*args)


def blackman_window(*args):
    op = _get_cache_prim(BlackmanWindow)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def bounding_box_decode(*args):
    op = _get_cache_prim(BoundingBoxDecode)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def bounding_box_encode(*args):
    op = _get_cache_prim(BoundingBoxEncode)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def broadcast(*args):
    op = _get_cache_prim(Broadcast)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def broadcast_to(*args):
    op = _get_cache_prim(BroadcastTo)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def bucketize(*args):
    op = _get_cache_prim(Bucketize)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def buffer_append(*args):
    op = _get_cache_prim(BufferAppend)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def buffer_get_item(*args):
    op = _get_cache_prim(BufferGetItem)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def buffer_sample(*args):
    op = _get_cache_prim(BufferSample)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def ctc_greedy_decoder(*args):
    op = _get_cache_prim(CTCGreedyDecoder)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def ctc_loss(*args):
    op = _get_cache_prim(CTCLoss)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def ctc_loss_v2(*args):
    op = _get_cache_prim(CTCLossV2)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


cast_op = Cast().set_device('GPU')
def cast(*args):
    return cast_op(*args)


def cauchy(*args):
    op = _get_cache_prim(Cauchy)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def cdist(*args):
    op = _get_cache_prim(Cdist)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def ce_lu(*args):
    op = _get_cache_prim(CeLU)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


ceil_op = Ceil().set_device('GPU')
def ceil(*args):
    return ceil_op(*args)


def channel_shuffle(*args):
    op = _get_cache_prim(ChannelShuffle)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


check_numerics_op = CheckNumerics().set_device('GPU')
def check_numerics(*args):
    return check_numerics_op(*args)


check_valid_op = CheckValid().set_device('GPU')
def check_valid(*args):
    return check_valid_op(*args)


def cholesky(*args):
    op = _get_cache_prim(Cholesky)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def cholesky_inverse(*args):
    op = _get_cache_prim(CholeskyInverse)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def cholesky_solve(*args):
    op = _get_cache_prim(CholeskySolve)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


coalesce_op = Coalesce().set_device('GPU')
def coalesce(*args):
    return coalesce_op(*args)


def col2_im(*args):
    op = _get_cache_prim(Col2Im)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def collective_gather(*args):
    op = _get_cache_prim(CollectiveGather)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def collective_scatter(*args):
    op = _get_cache_prim(CollectiveScatter)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def combined_non_max_suppression(*args):
    op = _get_cache_prim(CombinedNonMaxSuppression)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


compare_and_bitpack_op = CompareAndBitpack().set_device('GPU')
def compare_and_bitpack(*args):
    return compare_and_bitpack_op(*args)


complex_op = Complex().set_device('GPU')
def complex(*args):
    return complex_op(*args)


complex_abs_op = ComplexAbs().set_device('GPU')
def complex_abs(*args):
    return complex_abs_op(*args)


def compute_accidental_hits(*args):
    op = _get_cache_prim(ComputeAccidentalHits)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def concat(*args):
    op = _get_cache_prim(Concat)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def confusion_matrix(*args):
    op = _get_cache_prim(ConfusionMatrix)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


conj_op = Conj().set_device('GPU')
def conj(*args):
    return conj_op(*args)


conjugate_transpose_op = ConjugateTranspose().set_device('GPU')
def conjugate_transpose(*args):
    return conjugate_transpose_op(*args)


def conv2_d(*args):
    op = _get_cache_prim(Conv2D)(*args[-9:]).set_device('GPU')
    return op(*args[:-9])


def conv2_d_backprop_input(*args):
    op = _get_cache_prim(Conv2DBackpropInput)(*args[-10:]).set_device('GPU')
    return op(*args[:-10])


def conv2_d_transpose(*args):
    op = _get_cache_prim(Conv2DTranspose)(*args[-10:]).set_device('GPU')
    return op(*args[:-10])


def conv3_d(*args):
    op = _get_cache_prim(Conv3D)(*args[-9:]).set_device('GPU')
    return op(*args[:-9])


def conv3_d_transpose(*args):
    op = _get_cache_prim(Conv3DTranspose)(*args[-11:]).set_device('GPU')
    return op(*args[:-11])


copy_with_slice_op = CopyWithSlice().set_device('GPU')
def copy_with_slice(*args):
    return copy_with_slice_op(*args)


cos_op = Cos().set_device('GPU')
def cos(*args):
    return cos_op(*args)


cosh_op = Cosh().set_device('GPU')
def cosh(*args):
    return cosh_op(*args)


def count_non_zero(*args):
    op = _get_cache_prim(CountNonZero)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def crop_and_resize(*args):
    op = _get_cache_prim(CropAndResize)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def cross(*args):
    op = _get_cache_prim(Cross)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def cum_prod(*args):
    op = _get_cache_prim(CumProd)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def cum_sum(*args):
    op = _get_cache_prim(CumSum)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def cummax(*args):
    op = _get_cache_prim(Cummax)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def cummin(*args):
    op = _get_cache_prim(Cummin)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def cumulative_logsumexp(*args):
    op = _get_cache_prim(CumulativeLogsumexp)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


d_type_op = DType().set_device('GPU')
def d_type(*args):
    return d_type_op(*args)


def data_format_dim_map(*args):
    op = _get_cache_prim(DataFormatDimMap)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def data_format_vec_permute(*args):
    op = _get_cache_prim(DataFormatVecPermute)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def deformable_offsets(*args):
    op = _get_cache_prim(DeformableOffsets)(*args[-7:]).set_device('GPU')
    return op(*args[:-7])


dense_op = Dense().set_device('GPU')
def dense(*args):
    return dense_op(*args)


depend_op = Depend().set_device('GPU')
def depend(*args):
    return depend_op(*args)


def depth_to_space(*args):
    op = _get_cache_prim(DepthToSpace)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def depthwise_conv2d_native(*args):
    op = _get_cache_prim(DepthwiseConv2dNative)(*args[-8:]).set_device('GPU')
    return op(*args[:-8])


diag_op = Diag().set_device('GPU')
def diag(*args):
    return diag_op(*args)


diag_part_op = DiagPart().set_device('GPU')
def diag_part(*args):
    return diag_part_op(*args)


digamma_op = Digamma().set_device('GPU')
def digamma(*args):
    return digamma_op(*args)


def dilation2_d(*args):
    op = _get_cache_prim(Dilation2D)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


div_op = Div().set_device('GPU')
def div(*args):
    return div_op(*args)


div_no_nan_op = DivNoNan().set_device('GPU')
def div_no_nan(*args):
    return div_no_nan_op(*args)


def dropout(*args):
    op = _get_cache_prim(Dropout)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def dropout2_d(*args):
    op = _get_cache_prim(Dropout2D)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def dropout3_d(*args):
    op = _get_cache_prim(Dropout3D)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def dropout_gen_mask(*args):
    op = _get_cache_prim(DropoutGenMask)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def dynamic_gruv2(*args):
    op = _get_cache_prim(DynamicGRUV2)(*args[-10:]).set_device('GPU')
    return op(*args[:-10])


def dynamic_rnn(*args):
    op = _get_cache_prim(DynamicRNN)(*args[-11:]).set_device('GPU')
    return op(*args[:-11])


def dynamic_shape(*args):
    op = _get_cache_prim(DynamicShape)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def edit_distance(*args):
    op = _get_cache_prim(EditDistance)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def eig(*args):
    op = _get_cache_prim(Eig)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def einsum(*args):
    op = _get_cache_prim(Einsum)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def elu(*args):
    op = _get_cache_prim(Elu)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


embedding_lookup_op = EmbeddingLookup().set_device('GPU')
def embedding_lookup(*args):
    return embedding_lookup_op(*args)


eps_op = Eps().set_device('GPU')
def eps(*args):
    return eps_op(*args)


equal_op = Equal().set_device('GPU')
def equal(*args):
    return equal_op(*args)


equal_count_op = EqualCount().set_device('GPU')
def equal_count(*args):
    return equal_count_op(*args)


erf_op = Erf().set_device('GPU')
def erf(*args):
    return erf_op(*args)


erfc_op = Erfc().set_device('GPU')
def erfc(*args):
    return erfc_op(*args)


erfinv_op = Erfinv().set_device('GPU')
def erfinv(*args):
    return erfinv_op(*args)


erfinv_op = Erfinv().set_device('GPU')
def erfinv(*args):
    return erfinv_op(*args)


def euclidean_norm(*args):
    op = _get_cache_prim(EuclideanNorm)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


exp_op = Exp().set_device('GPU')
def exp(*args):
    return exp_op(*args)



expand_dims_op = ExpandDims().set_device('GPU')
def expand_dims(*args):
    return expand_dims_op(*args)


expm1_op = Expm1().set_device('GPU')
def expm1(*args):
    return expm1_op(*args)


def extract_glimpse(*args):
    op = _get_cache_prim(ExtractGlimpse)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def extract_image_patches(*args):
    op = _get_cache_prim(ExtractImagePatches)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def extract_volume_patches(*args):
    op = _get_cache_prim(ExtractVolumePatches)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


eye_op = Eye().set_device('GPU')
def eye(*args):
    return eye_op(*args)


def fft_with_size(*args):
    op = _get_cache_prim(FFTWithSize)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


fast_ge_lu_op = FastGeLU().set_device('GPU')
def fast_ge_lu(*args):
    return fast_ge_lu_op(*args)


fill_op = Fill().set_device('GPU')
def fill(*args):
    return fill_op(*args)


def fill_diagonal(*args):
    op = _get_cache_prim(FillDiagonal)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


fill_v2_op = FillV2().set_device('GPU')
def fill_v2(*args):
    return fill_v2_op(*args)


fills_op = Fills().set_device('GPU')
def fills(*args):
    return fills_op(*args)


flatten_op = Flatten().set_device('GPU')
def flatten(*args):
    return flatten_op(*args)


float_status_op = FloatStatus().set_device('GPU')
def float_status(*args):
    return float_status_op(*args)


floor_op = Floor().set_device('GPU')
def floor(*args):
    return floor_op(*args)


floor_div_op = FloorDiv().set_device('GPU')
def floor_div(*args):
    return floor_div_op(*args)


floor_mod_op = FloorMod().set_device('GPU')
def floor_mod(*args):
    return floor_mod_op(*args)


fmax_op = Fmax().set_device('GPU')
def fmax(*args):
    return fmax_op(*args)


fmin_op = Fmin().set_device('GPU')
def fmin(*args):
    return fmin_op(*args)


fori_loop_op = ForiLoop().set_device('GPU')
def fori_loop(*args):
    return fori_loop_op(*args)


def fractional_avg_pool(*args):
    op = _get_cache_prim(FractionalAvgPool)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def fractional_max_pool(*args):
    op = _get_cache_prim(FractionalMaxPool)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def fractional_max_pool3_d_with_fixed_ksize(*args):
    op = _get_cache_prim(FractionalMaxPool3DWithFixedKsize)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def fractional_max_pool_with_fixed_ksize(*args):
    op = _get_cache_prim(FractionalMaxPoolWithFixedKsize)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def fused_ada_factor(*args):
    op = _get_cache_prim(FusedAdaFactor)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def fused_ada_factor_with_global_norm(*args):
    op = _get_cache_prim(FusedAdaFactorWithGlobalNorm)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def fused_cast_adam_weight_decay(*args):
    op = _get_cache_prim(FusedCastAdamWeightDecay)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def fused_sparse_adam(*args):
    op = _get_cache_prim(FusedSparseAdam)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def fused_sparse_ftrl(*args):
    op = _get_cache_prim(FusedSparseFtrl)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


def fused_sparse_lazy_adam(*args):
    op = _get_cache_prim(FusedSparseLazyAdam)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def fused_sparse_proximal_adagrad(*args):
    op = _get_cache_prim(FusedSparseProximalAdagrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


fused_weight_scale_apply_momentum_op = FusedWeightScaleApplyMomentum().set_device('GPU')
def fused_weight_scale_apply_momentum(*args):
    return fused_weight_scale_apply_momentum_op(*args)


def glu(*args):
    op = _get_cache_prim(GLU)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def gamma(*args):
    op = _get_cache_prim(Gamma)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def gather(*args):
    op = _get_cache_prim(Gather)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


gather_d_op = GatherD().set_device('GPU')
def gather_d(*args):
    return gather_d_op(*args)


gather_nd_op = GatherNd().set_device('GPU')
def gather_nd(*args):
    return gather_nd_op(*args)


gcd_op = Gcd().set_device('GPU')
def gcd(*args):
    return gcd_op(*args)


ge_lu_op = GeLU().set_device('GPU')
def ge_lu(*args):
    return ge_lu_op(*args)


ge_switch_op = GeSwitch().set_device('GPU')
def ge_switch(*args):
    return ge_switch_op(*args)


geqrf_op = Geqrf().set_device('GPU')
def geqrf(*args):
    return geqrf_op(*args)


ger_op = Ger().set_device('GPU')
def ger(*args):
    return ger_op(*args)


def get_next(*args):
    op = _get_cache_prim(GetNext)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


greater_op = Greater().set_device('GPU')
def greater(*args):
    return greater_op(*args)


greater_equal_op = GreaterEqual().set_device('GPU')
def greater_equal(*args):
    return greater_equal_op(*args)


def grid_sampler2_d(*args):
    op = _get_cache_prim(GridSampler2D)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def grid_sampler3_d(*args):
    op = _get_cache_prim(GridSampler3D)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


group_topk_op = GroupTopk().set_device('GPU')
def group_topk(*args):
    return group_topk_op(*args)


hsv_to_rgb_op = HSVToRGB().set_device('GPU')
def hsv_to_rgb(*args):
    return hsv_to_rgb_op(*args)


def h_shrink(*args):
    op = _get_cache_prim(HShrink)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


h_sigmoid_op = HSigmoid().set_device('GPU')
def h_sigmoid(*args):
    return h_sigmoid_op(*args)


h_swish_op = HSwish().set_device('GPU')
def h_swish(*args):
    return h_swish_op(*args)


def hamming_window(*args):
    op = _get_cache_prim(HammingWindow)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


heaviside_op = Heaviside().set_device('GPU')
def heaviside(*args):
    return heaviside_op(*args)


def histogram(*args):
    op = _get_cache_prim(Histogram)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def histogram_fixed_width(*args):
    op = _get_cache_prim(HistogramFixedWidth)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


histogram_summary_op = HistogramSummary().set_device('GPU')
def histogram_summary(*args):
    return histogram_summary_op(*args)


def hook_backward(*args):
    op = _get_cache_prim(HookBackward)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


hypot_op = Hypot().set_device('GPU')
def hypot(*args):
    return hypot_op(*args)


def iou(*args):
    op = _get_cache_prim(IOU)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


identity_op = Identity().set_device('GPU')
def identity(*args):
    return identity_op(*args)


identity_n_op = IdentityN().set_device('GPU')
def identity_n(*args):
    return identity_n_op(*args)


igamma_op = Igamma().set_device('GPU')
def igamma(*args):
    return igamma_op(*args)


igammac_op = Igammac().set_device('GPU')
def igammac(*args):
    return igammac_op(*args)


def im2_col(*args):
    op = _get_cache_prim(Im2Col)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


imag_op = Imag().set_device('GPU')
def imag(*args):
    return imag_op(*args)


image_summary_op = ImageSummary().set_device('GPU')
def image_summary(*args):
    return image_summary_op(*args)


def in_top_k(*args):
    op = _get_cache_prim(InTopK)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def index_add(*args):
    op = _get_cache_prim(IndexAdd)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


index_fill_op = IndexFill().set_device('GPU')
def index_fill(*args):
    return index_fill_op(*args)


def index_put(*args):
    op = _get_cache_prim(IndexPut)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def inplace_add(*args):
    op = _get_cache_prim(InplaceAdd)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def inplace_index_add(*args):
    op = _get_cache_prim(InplaceIndexAdd)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def inplace_sub(*args):
    op = _get_cache_prim(InplaceSub)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def inplace_update(*args):
    op = _get_cache_prim(InplaceUpdate)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


inplace_update_v2_op = InplaceUpdateV2().set_device('GPU')
def inplace_update_v2(*args):
    return inplace_update_v2_op(*args)


def insert_gradient_of(*args):
    op = _get_cache_prim(InsertGradientOf)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


inv_op = Inv().set_device('GPU')
def inv(*args):
    return inv_op(*args)


invert_op = Invert().set_device('GPU')
def invert(*args):
    return invert_op(*args)


invert_permutation_op = InvertPermutation().set_device('GPU')
def invert_permutation(*args):
    return invert_permutation_op(*args)


def is_close(*args):
    op = _get_cache_prim(IsClose)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


is_finite_op = IsFinite().set_device('GPU')
def is_finite(*args):
    return is_finite_op(*args)


is_inf_op = IsInf().set_device('GPU')
def is_inf(*args):
    return is_inf_op(*args)


is_nan_op = IsNan().set_device('GPU')
def is_nan(*args):
    return is_nan_op(*args)


def kl_div_loss(*args):
    op = _get_cache_prim(KLDivLoss)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


l2_loss_op = L2Loss().set_device('GPU')
def l2_loss(*args):
    return l2_loss_op(*args)


def l2_normalize(*args):
    op = _get_cache_prim(L2Normalize)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def lars_update(*args):
    op = _get_cache_prim(LARSUpdate)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def lrn(*args):
    op = _get_cache_prim(LRN)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


def lstm(*args):
    op = _get_cache_prim(LSTM)(*args[-7:]).set_device('GPU')
    return op(*args[:-7])


def layer_norm(*args):
    op = _get_cache_prim(LayerNorm)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


lcm_op = Lcm().set_device('GPU')
def lcm(*args):
    return lcm_op(*args)


left_shift_op = LeftShift().set_device('GPU')
def left_shift(*args):
    return left_shift_op(*args)


lerp_op = Lerp().set_device('GPU')
def lerp(*args):
    return lerp_op(*args)


lerp_scalar_op = LerpScalar().set_device('GPU')
def lerp_scalar(*args):
    return lerp_scalar_op(*args)


less_op = Less().set_device('GPU')
def less(*args):
    return less_op(*args)


less_equal_op = LessEqual().set_device('GPU')
def less_equal(*args):
    return less_equal_op(*args)


lgamma_op = Lgamma().set_device('GPU')
def lgamma(*args):
    return lgamma_op(*args)


lin_space_op = LinSpace().set_device('GPU')
def lin_space(*args):
    return lin_space_op(*args)


def list_diff(*args):
    op = _get_cache_prim(ListDiff)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


log_op = Log().set_device('GPU')
def log(*args):
    return log_op(*args)


log1p_op = Log1p().set_device('GPU')
def log1p(*args):
    return log1p_op(*args)


log_matrix_determinant_op = LogMatrixDeterminant().set_device('GPU')
def log_matrix_determinant(*args):
    return log_matrix_determinant_op(*args)


def log_normal_reverse(*args):
    op = _get_cache_prim(LogNormalReverse)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def log_softmax(*args):
    op = _get_cache_prim(LogSoftmax)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


log_softmax_ext_op = LogSoftmaxExt().set_device('GPU')
def log_softmax_ext(*args):
    return log_softmax_ext_op(*args)


def log_space(*args):
    op = _get_cache_prim(LogSpace)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def log_uniform_candidate_sampler(*args):
    op = _get_cache_prim(LogUniformCandidateSampler)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


logical_and_op = LogicalAnd().set_device('GPU')
def logical_and(*args):
    return logical_and_op(*args)


logical_not_op = LogicalNot().set_device('GPU')
def logical_not(*args):
    return logical_not_op(*args)


logical_or_op = LogicalOr().set_device('GPU')
def logical_or(*args):
    return logical_or_op(*args)


logical_xor_op = LogicalXor().set_device('GPU')
def logical_xor(*args):
    return logical_xor_op(*args)


def logit(*args):
    op = _get_cache_prim(Logit)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def lower_bound(*args):
    op = _get_cache_prim(LowerBound)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def lp_norm(*args):
    op = _get_cache_prim(LpNorm)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def lstsq(*args):
    op = _get_cache_prim(Lstsq)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


lu_solve_op = LuSolve().set_device('GPU')
def lu_solve(*args):
    return lu_solve_op(*args)


def lu_unpack(*args):
    op = _get_cache_prim(LuUnpack)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


map_cache_idx_op = MapCacheIdx().set_device('GPU')
def map_cache_idx(*args):
    return map_cache_idx_op(*args)


map_uniform_op = MapUniform().set_device('GPU')
def map_uniform(*args):
    return map_uniform_op(*args)


masked_fill_op = MaskedFill().set_device('GPU')
def masked_fill(*args):
    return masked_fill_op(*args)


masked_scatter_op = MaskedScatter().set_device('GPU')
def masked_scatter(*args):
    return masked_scatter_op(*args)


masked_select_op = MaskedSelect().set_device('GPU')
def masked_select(*args):
    return masked_select_op(*args)


def mat_mul(*args):
    op = _get_cache_prim(MatMul)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


matrix_band_part_op = MatrixBandPart().set_device('GPU')
def matrix_band_part(*args):
    return matrix_band_part_op(*args)


matrix_determinant_op = MatrixDeterminant().set_device('GPU')
def matrix_determinant(*args):
    return matrix_determinant_op(*args)


def matrix_diag_part_v3(*args):
    op = _get_cache_prim(MatrixDiagPartV3)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def matrix_diag_v3(*args):
    op = _get_cache_prim(MatrixDiagV3)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


matrix_exp_op = MatrixExp().set_device('GPU')
def matrix_exp(*args):
    return matrix_exp_op(*args)


def matrix_inverse(*args):
    op = _get_cache_prim(MatrixInverse)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


matrix_logarithm_op = MatrixLogarithm().set_device('GPU')
def matrix_logarithm(*args):
    return matrix_logarithm_op(*args)


def matrix_power(*args):
    op = _get_cache_prim(MatrixPower)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def matrix_set_diag_v3(*args):
    op = _get_cache_prim(MatrixSetDiagV3)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def matrix_solve(*args):
    op = _get_cache_prim(MatrixSolve)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def matrix_solve_ls(*args):
    op = _get_cache_prim(MatrixSolveLs)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def matrix_triangular_solve(*args):
    op = _get_cache_prim(MatrixTriangularSolve)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def max_pool(*args):
    op = _get_cache_prim(MaxPool)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def max_pool3_d(*args):
    op = _get_cache_prim(MaxPool3D)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def max_pool3_d_with_argmax(*args):
    op = _get_cache_prim(MaxPool3DWithArgmax)(*args[-7:]).set_device('GPU')
    return op(*args[:-7])


def max_pool_with_argmax(*args):
    op = _get_cache_prim(MaxPoolWithArgmax)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def max_pool_with_argmax_v2(*args):
    op = _get_cache_prim(MaxPoolWithArgmaxV2)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def max_unpool2_d(*args):
    op = _get_cache_prim(MaxUnpool2D)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


def max_unpool3_d(*args):
    op = _get_cache_prim(MaxUnpool3D)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


maximum_op = Maximum().set_device('GPU')
def maximum(*args):
    return maximum_op(*args)


merge_op = Merge().set_device('GPU')
def merge(*args):
    return merge_op(*args)


def meshgrid(*args):
    op = _get_cache_prim(Meshgrid)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


minimum_op = Minimum().set_device('GPU')
def minimum(*args):
    return minimum_op(*args)


def mirror_pad(*args):
    op = _get_cache_prim(MirrorPad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


mish_op = Mish().set_device('GPU')
def mish(*args):
    return mish_op(*args)


mod_op = Mod().set_device('GPU')
def mod(*args):
    return mod_op(*args)


def morph(*args):
    op = _get_cache_prim(Morph)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


move_to_op = MoveTo().set_device('GPU')
def move_to(*args):
    return move_to_op(*args)


mul_op = Mul().set_device('GPU')
def mul(*args):
    return mul_op(*args)


mul_no_nan_op = MulNoNan().set_device('GPU')
def mul_no_nan(*args):
    return mul_no_nan_op(*args)


def multi_margin_loss(*args):
    op = _get_cache_prim(MultiMarginLoss)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def multilabel_margin_loss(*args):
    op = _get_cache_prim(MultilabelMarginLoss)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def multinomial(*args):
    op = _get_cache_prim(Multinomial)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def multinomial_with_replacement(*args):
    op = _get_cache_prim(MultinomialWithReplacement)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def mvlgamma(*args):
    op = _get_cache_prim(Mvlgamma)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def nll_loss(*args):
    op = _get_cache_prim(NLLLoss)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def nms_with_mask(*args):
    op = _get_cache_prim(NMSWithMask)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def nan_to_num(*args):
    op = _get_cache_prim(NanToNum)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


neg_op = Neg().set_device('GPU')
def neg(*args):
    return neg_op(*args)


def neighbor_exchange(*args):
    op = _get_cache_prim(NeighborExchange)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def neighbor_exchange_v2(*args):
    op = _get_cache_prim(NeighborExchangeV2)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


next_after_op = NextAfter().set_device('GPU')
def next_after(*args):
    return next_after_op(*args)


def no_repeat_n_gram(*args):
    op = _get_cache_prim(NoRepeatNGram)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def non_deterministic_ints(*args):
    op = _get_cache_prim(NonDeterministicInts)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


non_max_suppression_v3_op = NonMaxSuppressionV3().set_device('GPU')
def non_max_suppression_v3(*args):
    return non_max_suppression_v3_op(*args)


non_max_suppression_with_overlaps_op = NonMaxSuppressionWithOverlaps().set_device('GPU')
def non_max_suppression_with_overlaps(*args):
    return non_max_suppression_with_overlaps_op(*args)


non_zero_op = NonZero().set_device('GPU')
def non_zero(*args):
    return non_zero_op(*args)


not_equal_op = NotEqual().set_device('GPU')
def not_equal(*args):
    return not_equal_op(*args)


def nth_element(*args):
    op = _get_cache_prim(NthElement)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def nuclear_norm(*args):
    op = _get_cache_prim(NuclearNorm)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def one_hot(*args):
    op = _get_cache_prim(OneHot)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


ones_op = Ones().set_device('GPU')
def ones(*args):
    return ones_op(*args)


ones_like_op = OnesLike().set_device('GPU')
def ones_like(*args):
    return ones_like_op(*args)


orgqr_op = Orgqr().set_device('GPU')
def orgqr(*args):
    return orgqr_op(*args)


def ormqr(*args):
    op = _get_cache_prim(Ormqr)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


p_re_lu_op = PReLU().set_device('GPU')
def p_re_lu(*args):
    return p_re_lu_op(*args)


def pack(*args):
    op = _get_cache_prim(Pack)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def pad(*args):
    op = _get_cache_prim(Pad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def pad_v3(*args):
    op = _get_cache_prim(PadV3)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def padding(*args):
    op = _get_cache_prim(Padding)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def paged_attention(*args):
    op = _get_cache_prim(PagedAttention)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def paged_attention_mask(*args):
    op = _get_cache_prim(PagedAttentionMask)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


parallel_concat_op = ParallelConcat().set_device('GPU')
def parallel_concat(*args):
    return parallel_concat_op(*args)


def parameterized_truncated_normal(*args):
    op = _get_cache_prim(ParameterizedTruncatedNormal)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


partial_op = Partial().set_device('GPU')
def partial(*args):
    return partial_op(*args)


def pdist(*args):
    op = _get_cache_prim(Pdist)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def poisson(*args):
    op = _get_cache_prim(Poisson)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


polar_op = Polar().set_device('GPU')
def polar(*args):
    return polar_op(*args)


polygamma_op = Polygamma().set_device('GPU')
def polygamma(*args):
    return polygamma_op(*args)


population_count_op = PopulationCount().set_device('GPU')
def population_count(*args):
    return population_count_op(*args)


pow_op = Pow().set_device('GPU')
def pow(*args):
    return pow_op(*args)


pull_op = Pull().set_device('GPU')
def pull(*args):
    return pull_op(*args)


def push(*args):
    op = _get_cache_prim(Push)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


py_execute_op = PyExecute().set_device('GPU')
def py_execute(*args):
    return py_execute_op(*args)


def py_func(*args):
    op = _get_cache_prim(PyFunc)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def qr(*args):
    op = _get_cache_prim(Qr)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def quantile(*args):
    op = _get_cache_prim(Quantile)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


rgb_to_hsv_op = RGBToHSV().set_device('GPU')
def rgb_to_hsv(*args):
    return rgb_to_hsv_op(*args)


def rnnt_loss(*args):
    op = _get_cache_prim(RNNTLoss)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def roi_align(*args):
    op = _get_cache_prim(ROIAlign)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


def ragged_range(*args):
    op = _get_cache_prim(RaggedRange)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def random_categorical(*args):
    op = _get_cache_prim(RandomCategorical)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def random_choice_with_mask(*args):
    op = _get_cache_prim(RandomChoiceWithMask)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def random_gamma(*args):
    op = _get_cache_prim(RandomGamma)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def random_gamma(*args):
    op = _get_cache_prim(RandomGamma)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def random_poisson(*args):
    op = _get_cache_prim(RandomPoisson)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def random_shuffle(*args):
    op = _get_cache_prim(RandomShuffle)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def randperm(*args):
    op = _get_cache_prim(Randperm)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def randperm_v2(*args):
    op = _get_cache_prim(RandpermV2)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def range(*args):
    op = _get_cache_prim(Range)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


rank_op = Rank().set_device('GPU')
def rank(*args):
    return rank_op(*args)


re_lu_op = ReLU().set_device('GPU')
def re_lu(*args):
    return re_lu_op(*args)


re_lu6_op = ReLU6().set_device('GPU')
def re_lu6(*args):
    return re_lu6_op(*args)


real_op = Real().set_device('GPU')
def real(*args):
    return real_op(*args)


real_div_op = RealDiv().set_device('GPU')
def real_div(*args):
    return real_div_op(*args)


def receive(*args):
    op = _get_cache_prim(Receive)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


reciprocal_op = Reciprocal().set_device('GPU')
def reciprocal(*args):
    return reciprocal_op(*args)


def reduce(*args):
    op = _get_cache_prim(Reduce)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def reduce_all(*args):
    op = _get_cache_prim(ReduceAll)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def reduce_any(*args):
    op = _get_cache_prim(ReduceAny)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def reduce_max(*args):
    op = _get_cache_prim(ReduceMax)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def reduce_mean(*args):
    op = _get_cache_prim(ReduceMean)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def reduce_min(*args):
    op = _get_cache_prim(ReduceMin)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def reduce_prod(*args):
    op = _get_cache_prim(ReduceProd)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def reduce_scatter(*args):
    op = _get_cache_prim(ReduceScatter)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def reduce_std(*args):
    op = _get_cache_prim(ReduceStd)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def reduce_sum(*args):
    op = _get_cache_prim(ReduceSum)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def renorm(*args):
    op = _get_cache_prim(Renorm)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


reshape_op = Reshape().set_device('GPU')
def reshape(*args):
    return reshape_op(*args)


reshape_and_cache_op = ReshapeAndCache().set_device('GPU')
def reshape_and_cache(*args):
    return reshape_and_cache_op(*args)


def reshard(*args):
    op = _get_cache_prim(Reshard)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def resize_area(*args):
    op = _get_cache_prim(ResizeArea)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def resize_bicubic(*args):
    op = _get_cache_prim(ResizeBicubic)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def resize_bilinear_v2(*args):
    op = _get_cache_prim(ResizeBilinearV2)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def resize_linear1_d(*args):
    op = _get_cache_prim(ResizeLinear1D)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def resize_nearest_neighbor(*args):
    op = _get_cache_prim(ResizeNearestNeighbor)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def resize_nearest_neighbor_v2(*args):
    op = _get_cache_prim(ResizeNearestNeighborV2)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


reusing_op = Reusing().set_device('GPU')
def reusing(*args):
    return reusing_op(*args)


def reverse_sequence(*args):
    op = _get_cache_prim(ReverseSequence)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def reverse_v2(*args):
    op = _get_cache_prim(ReverseV2)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


right_shift_op = RightShift().set_device('GPU')
def right_shift(*args):
    return right_shift_op(*args)


rint_op = Rint().set_device('GPU')
def rint(*args):
    return rint_op(*args)


def rms_norm(*args):
    op = _get_cache_prim(RmsNorm)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def roll(*args):
    op = _get_cache_prim(Roll)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


round_op = Round().set_device('GPU')
def round(*args):
    return round_op(*args)


rsqrt_op = Rsqrt().set_device('GPU')
def rsqrt(*args):
    return rsqrt_op(*args)


def sgd(*args):
    op = _get_cache_prim(SGD)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def stft(*args):
    op = _get_cache_prim(STFT)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def sample_distorted_bounding_box_v2(*args):
    op = _get_cache_prim(SampleDistortedBoundingBoxV2)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


scalar_summary_op = ScalarSummary().set_device('GPU')
def scalar_summary(*args):
    return scalar_summary_op(*args)


scalar_to_tensor_op = ScalarToTensor().set_device('GPU')
def scalar_to_tensor(*args):
    return scalar_to_tensor_op(*args)


def scale_and_translate(*args):
    op = _get_cache_prim(ScaleAndTranslate)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


scan_op = Scan().set_device('GPU')
def scan(*args):
    return scan_op(*args)


def scatter_add(*args):
    op = _get_cache_prim(ScatterAdd)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_add_with_axis(*args):
    op = _get_cache_prim(ScatterAddWithAxis)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_div(*args):
    op = _get_cache_prim(ScatterDiv)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_max(*args):
    op = _get_cache_prim(ScatterMax)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_min(*args):
    op = _get_cache_prim(ScatterMin)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_mul(*args):
    op = _get_cache_prim(ScatterMul)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


scatter_nd_op = ScatterNd().set_device('GPU')
def scatter_nd(*args):
    return scatter_nd_op(*args)


def scatter_nd_add(*args):
    op = _get_cache_prim(ScatterNdAdd)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_nd_div(*args):
    op = _get_cache_prim(ScatterNdDiv)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_nd_max(*args):
    op = _get_cache_prim(ScatterNdMax)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_nd_min(*args):
    op = _get_cache_prim(ScatterNdMin)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_nd_mul(*args):
    op = _get_cache_prim(ScatterNdMul)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_nd_sub(*args):
    op = _get_cache_prim(ScatterNdSub)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_nd_update(*args):
    op = _get_cache_prim(ScatterNdUpdate)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_sub(*args):
    op = _get_cache_prim(ScatterSub)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def scatter_update(*args):
    op = _get_cache_prim(ScatterUpdate)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


se_lu_op = SeLU().set_device('GPU')
def se_lu(*args):
    return se_lu_op(*args)


def search_sorted(*args):
    op = _get_cache_prim(SearchSorted)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


segment_max_op = SegmentMax().set_device('GPU')
def segment_max(*args):
    return segment_max_op(*args)


segment_mean_op = SegmentMean().set_device('GPU')
def segment_mean(*args):
    return segment_mean_op(*args)


segment_min_op = SegmentMin().set_device('GPU')
def segment_min(*args):
    return segment_min_op(*args)


segment_prod_op = SegmentProd().set_device('GPU')
def segment_prod(*args):
    return segment_prod_op(*args)


segment_sum_op = SegmentSum().set_device('GPU')
def segment_sum(*args):
    return segment_sum_op(*args)


select_op = Select().set_device('GPU')
def select(*args):
    return select_op(*args)


select_view_op = SelectView().set_device('GPU')
def select_view(*args):
    return select_view_op(*args)


def send(*args):
    op = _get_cache_prim(Send)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


shape_op = Shape().set_device('GPU')
def shape(*args):
    return shape_op(*args)


sigmoid_op = Sigmoid().set_device('GPU')
def sigmoid(*args):
    return sigmoid_op(*args)


sigmoid_cross_entropy_with_logits_op = SigmoidCrossEntropyWithLogits().set_device('GPU')
def sigmoid_cross_entropy_with_logits(*args):
    return sigmoid_cross_entropy_with_logits_op(*args)


sign_op = Sign().set_device('GPU')
def sign(*args):
    return sign_op(*args)


sin_op = Sin().set_device('GPU')
def sin(*args):
    return sin_op(*args)


sinc_op = Sinc().set_device('GPU')
def sinc(*args):
    return sinc_op(*args)


sinh_op = Sinh().set_device('GPU')
def sinh(*args):
    return sinh_op(*args)


size_op = Size().set_device('GPU')
def size(*args):
    return size_op(*args)


slice_op = Slice().set_device('GPU')
def slice(*args):
    return slice_op(*args)


def smooth_l1_loss(*args):
    op = _get_cache_prim(SmoothL1Loss)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def soft_margin_loss(*args):
    op = _get_cache_prim(SoftMarginLoss)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def soft_shrink(*args):
    op = _get_cache_prim(SoftShrink)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def softmax(*args):
    op = _get_cache_prim(Softmax)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


softmax_cross_entropy_with_logits_op = SoftmaxCrossEntropyWithLogits().set_device('GPU')
def softmax_cross_entropy_with_logits(*args):
    return softmax_cross_entropy_with_logits_op(*args)


softplus_op = Softplus().set_device('GPU')
def softplus(*args):
    return softplus_op(*args)


softsign_op = Softsign().set_device('GPU')
def softsign(*args):
    return softsign_op(*args)


def sort(*args):
    op = _get_cache_prim(Sort)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def space_to_batch(*args):
    op = _get_cache_prim(SpaceToBatch)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def space_to_batch_nd(*args):
    op = _get_cache_prim(SpaceToBatchND)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def space_to_depth(*args):
    op = _get_cache_prim(SpaceToDepth)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def sparse_apply_adadelta(*args):
    op = _get_cache_prim(SparseApplyAdadelta)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def sparse_apply_adagrad(*args):
    op = _get_cache_prim(SparseApplyAdagrad)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


def sparse_apply_adagrad_v2(*args):
    op = _get_cache_prim(SparseApplyAdagradV2)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def sparse_apply_ftrl(*args):
    op = _get_cache_prim(SparseApplyFtrl)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


def sparse_apply_ftrl_v2(*args):
    op = _get_cache_prim(SparseApplyFtrlV2)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def sparse_apply_proximal_adagrad(*args):
    op = _get_cache_prim(SparseApplyProximalAdagrad)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def sparse_apply_rms_prop(*args):
    op = _get_cache_prim(SparseApplyRMSProp)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


sparse_gather_v2_op = SparseGatherV2().set_device('GPU')
def sparse_gather_v2(*args):
    return sparse_gather_v2_op(*args)


sparse_slice_op = SparseSlice().set_device('GPU')
def sparse_slice(*args):
    return sparse_slice_op(*args)


def sparse_softmax_cross_entropy_with_logits(*args):
    op = _get_cache_prim(SparseSoftmaxCrossEntropyWithLogits)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


sparse_tensor_dense_add_op = SparseTensorDenseAdd().set_device('GPU')
def sparse_tensor_dense_add(*args):
    return sparse_tensor_dense_add_op(*args)


def sparse_tensor_dense_matmul(*args):
    op = _get_cache_prim(SparseTensorDenseMatmul)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


sparse_to_dense_op = SparseToDense().set_device('GPU')
def sparse_to_dense(*args):
    return sparse_to_dense_op(*args)


def split(*args):
    op = _get_cache_prim(Split)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def split_v(*args):
    op = _get_cache_prim(SplitV)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


sqrt_op = Sqrt().set_device('GPU')
def sqrt(*args):
    return sqrt_op(*args)


square_op = Square().set_device('GPU')
def square(*args):
    return square_op(*args)


square_sum_all_op = SquareSumAll().set_device('GPU')
def square_sum_all(*args):
    return square_sum_all_op(*args)


squared_difference_op = SquaredDifference().set_device('GPU')
def squared_difference(*args):
    return squared_difference_op(*args)


def squeeze(*args):
    op = _get_cache_prim(Squeeze)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def stack(*args):
    op = _get_cache_prim(Stack)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def standard_laplace(*args):
    op = _get_cache_prim(StandardLaplace)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def standard_normal(*args):
    op = _get_cache_prim(StandardNormal)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


stop_gradient_op = StopGradient().set_device('GPU')
def stop_gradient(*args):
    return stop_gradient_op(*args)


def strided_slice(*args):
    op = _get_cache_prim(StridedSlice)(*args[-5:]).set_device('GPU')
    return op(*args[:-5])


sub_op = Sub().set_device('GPU')
def sub(*args):
    return sub_op(*args)


sub_and_filter_op = SubAndFilter().set_device('GPU')
def sub_and_filter(*args):
    return sub_and_filter_op(*args)


def svd(*args):
    op = _get_cache_prim(Svd)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


tan_op = Tan().set_device('GPU')
def tan(*args):
    return tan_op(*args)


tanh_op = Tanh().set_device('GPU')
def tanh(*args):
    return tanh_op(*args)


def tensor_dump(*args):
    op = _get_cache_prim(TensorDump)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


tensor_scatter_add_op = TensorScatterAdd().set_device('GPU')
def tensor_scatter_add(*args):
    return tensor_scatter_add_op(*args)


tensor_scatter_div_op = TensorScatterDiv().set_device('GPU')
def tensor_scatter_div(*args):
    return tensor_scatter_div_op(*args)


def tensor_scatter_elements(*args):
    op = _get_cache_prim(TensorScatterElements)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


tensor_scatter_max_op = TensorScatterMax().set_device('GPU')
def tensor_scatter_max(*args):
    return tensor_scatter_max_op(*args)


tensor_scatter_min_op = TensorScatterMin().set_device('GPU')
def tensor_scatter_min(*args):
    return tensor_scatter_min_op(*args)


tensor_scatter_mul_op = TensorScatterMul().set_device('GPU')
def tensor_scatter_mul(*args):
    return tensor_scatter_mul_op(*args)


tensor_scatter_sub_op = TensorScatterSub().set_device('GPU')
def tensor_scatter_sub(*args):
    return tensor_scatter_sub_op(*args)


tensor_scatter_update_op = TensorScatterUpdate().set_device('GPU')
def tensor_scatter_update(*args):
    return tensor_scatter_update_op(*args)


tensor_shape_op = TensorShape().set_device('GPU')
def tensor_shape(*args):
    return tensor_shape_op(*args)


tensor_summary_op = TensorSummary().set_device('GPU')
def tensor_summary(*args):
    return tensor_summary_op(*args)


tile_op = Tile().set_device('GPU')
def tile(*args):
    return tile_op(*args)


def top_k(*args):
    op = _get_cache_prim(TopK)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


trace_op = Trace().set_device('GPU')
def trace(*args):
    return trace_op(*args)


transpose_op = Transpose().set_device('GPU')
def transpose(*args):
    return transpose_op(*args)


transpose_ext_view_op = TransposeExtView().set_device('GPU')
def transpose_ext_view(*args):
    return transpose_ext_view_op(*args)


transpose_view_op = TransposeView().set_device('GPU')
def transpose_view(*args):
    return transpose_view_op(*args)


tridiagonal_mat_mul_op = TridiagonalMatMul().set_device('GPU')
def tridiagonal_mat_mul(*args):
    return tridiagonal_mat_mul_op(*args)


def tril(*args):
    op = _get_cache_prim(Tril)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def tril_indices(*args):
    op = _get_cache_prim(TrilIndices)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def triplet_margin_loss(*args):
    op = _get_cache_prim(TripletMarginLoss)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


def triu(*args):
    op = _get_cache_prim(Triu)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


def triu_indices(*args):
    op = _get_cache_prim(TriuIndices)(*args[-4:]).set_device('GPU')
    return op(*args[:-4])


trunc_op = Trunc().set_device('GPU')
def trunc(*args):
    return trunc_op(*args)


truncate_div_op = TruncateDiv().set_device('GPU')
def truncate_div(*args):
    return truncate_div_op(*args)


truncate_mod_op = TruncateMod().set_device('GPU')
def truncate_mod(*args):
    return truncate_mod_op(*args)


def truncated_normal(*args):
    op = _get_cache_prim(TruncatedNormal)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])


tuple_to_array_op = TupleToArray().set_device('GPU')
def tuple_to_array(*args):
    return tuple_to_array_op(*args)


def uniform_candidate_sampler(*args):
    op = _get_cache_prim(UniformCandidateSampler)(*args[-6:]).set_device('GPU')
    return op(*args[:-6])


def uniform_int(*args):
    op = _get_cache_prim(UniformInt)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


def uniform_real(*args):
    op = _get_cache_prim(UniformReal)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


unique_op = Unique().set_device('GPU')
def unique(*args):
    return unique_op(*args)


def unique_consecutive(*args):
    op = _get_cache_prim(UniqueConsecutive)(*args[-3:]).set_device('GPU')
    return op(*args[:-3])



def unpack(*args):
    op = _get_cache_prim(Unpack)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


unravel_index_op = UnravelIndex().set_device('GPU')
def unravel_index(*args):
    return unravel_index_op(*args)


unsorted_segment_max_op = UnsortedSegmentMax().set_device('GPU')
def unsorted_segment_max(*args):
    return unsorted_segment_max_op(*args)


unsorted_segment_min_op = UnsortedSegmentMin().set_device('GPU')
def unsorted_segment_min(*args):
    return unsorted_segment_min_op(*args)


unsorted_segment_prod_op = UnsortedSegmentProd().set_device('GPU')
def unsorted_segment_prod(*args):
    return unsorted_segment_prod_op(*args)


unsorted_segment_sum_op = UnsortedSegmentSum().set_device('GPU')
def unsorted_segment_sum(*args):
    return unsorted_segment_sum_op(*args)


def unstack(*args):
    op = _get_cache_prim(Unstack)(*args[-2:]).set_device('GPU')
    return op(*args[:-2])


update_state_op = UpdateState().set_device('GPU')
def update_state(*args):
    return update_state_op(*args)


def upper_bound(*args):
    op = _get_cache_prim(UpperBound)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


upsample_nearest3_d_op = UpsampleNearest3D().set_device('GPU')
def upsample_nearest3_d(*args):
    return upsample_nearest3_d_op(*args)


def upsample_trilinear3_d(*args):
    op = _get_cache_prim(UpsampleTrilinear3D)(*args[-1:]).set_device('GPU')
    return op(*args[:-1])


while_loop_op = WhileLoop().set_device('GPU')
def while_loop(*args):
    return while_loop_op(*args)


xdivy_op = Xdivy().set_device('GPU')
def xdivy(*args):
    return xdivy_op(*args)


xlogy_op = Xlogy().set_device('GPU')
def xlogy(*args):
    return xlogy_op(*args)


zeros_op = Zeros().set_device('GPU')
def zeros(*args):
    return zeros_op(*args)


zeros_like_op = ZerosLike().set_device('GPU')
def zeros_like(*args):
    return zeros_like_op(*args)


zeta_op = Zeta().set_device('GPU')
def zeta(*args):
    return zeta_op(*args)
