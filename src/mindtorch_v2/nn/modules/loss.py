from ..module import Module
from .. import functional as F


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class CrossEntropyLoss(_Loss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', label_smoothing=0.0):
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)


class MSELoss(_Loss):
    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)


class BCELoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target, weight=self.weight,
                                                  reduction=self.reduction, pos_weight=self.pos_weight)


class NLLLoss(_Loss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.nll_loss(input, target, weight=self.weight,
                          ignore_index=self.ignore_index, reduction=self.reduction)


class L1Loss(_Loss):
    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', beta=1.0):
        super().__init__(size_average, reduce, reduction)
        self.beta = beta

    def forward(self, input, target):
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)


class KLDivLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', log_target=False):
        super().__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input, target):
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)


class HuberLoss(_Loss):
    def __init__(self, reduction='mean', delta=1.0):
        super().__init__(reduction=reduction)
        self.delta = delta

    def forward(self, input, target):
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)


class CosineEmbeddingLoss(_Loss):
    def __init__(self, margin=0, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1, input2, target):
        return F.cosine_embedding_loss(input1, input2, target, margin=self.margin,
                                       reduction=self.reduction)


class MarginRankingLoss(_Loss):
    def __init__(self, margin=0, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1, input2, target):
        return F.margin_ranking_loss(input1, input2, target, margin=self.margin,
                                     reduction=self.reduction)


class TripletMarginLoss(_Loss):
    def __init__(self, margin=1.0, p=2, eps=1e-6, swap=False,
                 size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, margin=self.margin,
                                     p=self.p, eps=self.eps, swap=self.swap,
                                     reduction=self.reduction)


class HingeEmbeddingLoss(_Loss):
    def __init__(self, margin=1.0, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input, target):
        return F.hinge_embedding_loss(input, target, margin=self.margin,
                                      reduction=self.reduction)


class SoftMarginLoss(_Loss):
    def forward(self, input, target):
        return F.soft_margin_loss(input, target, reduction=self.reduction)


class MultiMarginLoss(_Loss):
    def __init__(self, p=1, margin=1.0, weight=None, size_average=None,
                 reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.p = p
        self.margin = margin
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return F.multi_margin_loss(input, target, p=self.p, margin=self.margin,
                                   weight=self.weight, reduction=self.reduction)


class MultiLabelSoftMarginLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return F.multilabel_soft_margin_loss(input, target, weight=self.weight,
                                             reduction=self.reduction)


class PoissonNLLLoss(_Loss):
    def __init__(self, log_input=True, full=False, size_average=None, eps=1e-8,
                 reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    def forward(self, log_input, target):
        return F.poisson_nll_loss(log_input, target, log_input=self.log_input,
                                  full=self.full, eps=self.eps, reduction=self.reduction)


class CTCLoss(_Loss):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        super().__init__(reduction=reduction)
        self.blank = blank
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                          blank=self.blank, reduction=self.reduction,
                          zero_infinity=self.zero_infinity)


class GaussianNLLLoss(_Loss):
    def __init__(self, full=False, eps=1e-6, reduction='mean'):
        super().__init__(reduction=reduction)
        self.full = full
        self.eps = eps

    def forward(self, input, target, var):
        return F.gaussian_nll_loss(input, target, var, full=self.full,
                                   eps=self.eps, reduction=self.reduction)
