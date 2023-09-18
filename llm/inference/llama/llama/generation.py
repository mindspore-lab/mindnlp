# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import mindspore
from mindspore import ops, Tensor, mutable
from mindspore.ops._tracefunc import trace

from .tokenizer import Tokenizer
from .model import Transformer

import time

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        config = self.model.config
        assert bsz <= config.max_batch_size, (bsz, config.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(config.max_seq_len, max_gen_len + max_prompt_size)

        tokens = ops.full((bsz, total_len), self.tokenizer.pad_id, dtype=mindspore.int32)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = mindspore.Tensor(t, dtype=mindspore.int32)
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            s = time.time()
            logits = self.model(tokens[:, prev_pos:cur_pos], prev_pos)
            t = time.time()
            print(t - s)

            if temperature > 0:
                probs = ops.softmax(logits / temperature, axis=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = ops.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = ops.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.asnumpy().tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


@trace
def sample_top_p(probs, p):
    probs_sort, probs_idx = ops.sort(probs, axis=-1, descending=True)
    probs_sum = ops.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    # probs_sort[mask] = 0.0
    probs_sort = probs_sort * (1 - mask)
    probs_sort = ops.div(probs_sort, probs_sort.sum(axis=-1, keepdims=True))
    next_token = ops.multinomial(probs_sort, num_samples=1, replacement=False)
    next_token = ops.gather_elements(probs_idx, -1, next_token)
    return next_token
