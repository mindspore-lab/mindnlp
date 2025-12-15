import mindtorch
from mindtorch import nn


class Sampler(nn.Module):
    def forward(self, logits: mindtorch.Tensor, temperatures: mindtorch.Tensor):
        logits = logits.to(mindtorch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = mindtorch.softmax(logits, dim=-1, dtype=mindtorch.float)
        # logprobs = mindtorch.log_softmax(logits, dim=-1, dtype=mindtorch.float)
        epsilon = 1e-10
        sample_tokens = probs.div_(mindtorch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)
        return mindtorch.where(temperatures == 0, greedy_tokens, sample_tokens)
