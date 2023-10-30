# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=C0412

"""crf module"""

import mindspore
from mindspore import nn, ops, Tensor
from mindspore import Parameter
from mindspore.common.initializer import initializer, Uniform
from mindnlp.utils import less_min_pynative_first
if less_min_pynative_first:
    from mindnlp._legacy.functional import full, arange, where
else:
    from mindspore.ops import full, arange, where

def sequence_mask(seq_length, max_length, batch_first=False):
    """generate mask matrix by seq_length"""
    range_vector = arange(0, max_length, 1, dtype=seq_length.dtype)
    result = range_vector < seq_length.view(seq_length.shape + (1,))
    if batch_first:
        return result
    return result.swapaxes(0, 1)

class CRF(nn.Cell):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
        reduction: Specifies  the reduction to apply to the output:
            ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
            ``sum``: the output will be summed over batches. ``mean``: the output will be
            averaged over batches. ``token_mean``: the output will be averaged over tokens.

    Attributes:
        start_transitions (`~Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False, reduction: str = 'sum') -> None:
        super().__init__()
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.reduction = reduction
        self.start_transitions = Parameter(initializer(Uniform(0.1), (num_tags,)),
                                           name='start_transitions')
        self.end_transitions = Parameter(initializer(Uniform(0.1), (num_tags,)),
                                         name='end_transitions')
        self.transitions = Parameter(initializer(Uniform(0.1), (num_tags, num_tags)),
                                     name='transitions')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def construct(self, emissions, tags=None, seq_length=None):
        if tags is None:
            return self._decode(emissions, seq_length)
        return self._construct(emissions, tags, seq_length)

    def _construct(self, emissions, tags=None, seq_length=None):
        if self.batch_first:
            batch_size, max_length = tags.shape
            emissions = emissions.swapaxes(0, 1)
            tags = tags.swapaxes(0, 1)
        else:
            max_length, batch_size = tags.shape

        if seq_length is None:
            seq_length = full((batch_size,), max_length, dtype=mindspore.int64)

        mask = sequence_mask(seq_length, max_length)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, seq_length-1, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = denominator - numerator

        if self.reduction == 'none':
            return llh
        if self.reduction == 'sum':
            return llh.sum()
        if self.reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.astype(emissions.dtype).sum()

    def _decode(self, emissions, seq_length=None):
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        # self._validate(emissions, mask=mask)
        if self.batch_first:
            batch_size, max_length = emissions.shape[:2]
            emissions = emissions.swapaxes(0, 1)
        else:
            max_length, batch_size = emissions.shape[:2]

        if seq_length is None:
            seq_length = full((batch_size,), max_length, dtype=mindspore.int64)

        mask = sequence_mask(seq_length, max_length)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, seq_ends, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)

        seq_length, batch_size = tags.shape
        mask = mask.astype(emissions.dtype)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, arange(batch_size), tags[0]]

        i = Tensor(1, mindspore.int64)
        while i < seq_length:
        # for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, ops.arange(batch_size), tags[i]] * mask[i]
            i += 1

        # End transition score
        # shape: (batch_size,)
        last_tags = tags[seq_ends, arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_length = emissions.shape[0]
        mask = mask.astype(emissions.dtype)
        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        i = Tensor(1, mindspore.int32)
        while i < seq_length:
        # for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.expand_dims(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].expand_dims(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = ops.logsumexp(next_score, axis=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = where(mask[i].astype(mindspore.bool_).expand_dims(1), next_score, score)
            i += 1

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return ops.logsumexp(score, axis=1)

    def _viterbi_decode(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_length = mask.shape[0]

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = ()

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.expand_dims(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].expand_dims(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            indices = next_score.argmax(axis=1)
            next_score = next_score.max(axis=1)
            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = where(mask[i].expand_dims(1), next_score, score)
            history += (indices,)
        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        return score, history

    @staticmethod
    def post_decode(score, history, seq_length):
        """Trace back the best tag sequence based on the score and history tensors."""
        # Now, compute the best path for each sample
        batch_size = seq_length.shape[0]
        seq_ends = seq_length - 1
        # shape: (batch_size,)
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            best_last_tag = score[idx].argmax(axis=0)
            best_tags = [best_last_tag]
            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag)
            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

__all__ = ["CRF", "sequence_mask"]
