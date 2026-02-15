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
