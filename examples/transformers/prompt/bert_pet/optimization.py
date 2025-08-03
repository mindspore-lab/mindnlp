import mindspore
from mindspore import ops, nn
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

class WarmUpPolynomialDecayLR(LearningRateSchedule):
    """"""
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.end_learning_rate = end_learning_rate
        self.decay_steps = decay_steps
        self.power = power
    
    def construct(self, global_step):
        # warmup lr
        warmup_percent = global_step.astype(mindspore.float32) / self.warmup_steps
        warmup_learning_rate = self.learning_rate * warmup_percent
        # polynomial lr
        global_step = ops.minimum(global_step, self.decay_steps)
        decayed_learning_rate = (self.learning_rate - self.end_learning_rate) * \
                                ops.pow((self.decay_steps*1.0 - global_step*1.0) / (self.decay_steps*1.0-self.warmup_steps*1.0), self.power) + \
                                self.end_learning_rate
        is_warmup = (global_step < self.warmup_steps).astype(mindspore.float32)
        learning_rate = ((1.0 - is_warmup) * decayed_learning_rate + is_warmup * warmup_learning_rate)
        return learning_rate

def create_optimizer(model, init_lr, num_train_steps, num_warmup_steps):
    lr = WarmUpPolynomialDecayLR(init_lr, 0.0, num_warmup_steps, num_train_steps, 1.0)
    optim_params = list(filter(lambda x: 'gamma' not in x.name \
                                        and 'beta' not in x.name \
                                        and 'bias' not in x.name,
                               model.trainable_params()))
    group_params = [{'params': optim_params, 'weight_decay': 0.01}]
    optim = nn.AdamWeightDecay(group_params, lr)
    return optim