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

"""crf module"""

import mindspore
from mindspore import nn, ops, Tensor
from mindspore import Parameter
from mindspore.common.initializer import initializer, Uniform
from mindnlp._legacy.functional import full, arange, where

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
        r"""
        Initializes an instance of the CRF (Conditional Random Field) class.
        
        Args:
            self: The CRF object itself.
            num_tags (int): The number of tags in the CRF model. Must be a positive integer.
            batch_first (bool, optional): Whether the input tensors are provided in batch-first format. Defaults to False.
            reduction (str, optional): The reduction method for aggregating the loss. Must be one of 'none', 'sum', 'mean', or 'token_mean'. Defaults to 'sum'.
        
        Returns:
            None
        
        Raises:
            ValueError: If the number of tags is zero or negative.
            ValueError: If the reduction method is not one of 'none', 'sum', 'mean', or 'token_mean'.
        """
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
        r"""
        Return a string representation of the CRF object.
        
        Args:
            self: The CRF object itself.
        
        Returns:
            A string representation of the CRF object. The string includes the class name and the value of the 'num_tags' attribute.
        
        Raises:
            None.
        """
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def construct(self, emissions, tags=None, seq_length=None):
        if tags is None:
            return self._decode(emissions, seq_length)
        return self._construct(emissions, tags, seq_length)

    def _construct(self, emissions, tags=None, seq_length=None):
        r"""
        This method '_construct' in the class 'CRF' is responsible for constructing the conditional random field (CRF) based on the provided emissions, tags, and sequence length.
        
        Args:
            self (object): The instance of the CRF class.
            emissions (array-like): The emissions for the CRF, representing the observed features. It should be a tensor of shape (max_length, batch_size, num_labels).
            tags (array-like, optional): The tags for the CRF, representing the sequence of labels. It should be a tensor of shape (max_length, batch_size) if self.batch_first is True, otherwise (batch_size,
max_length).
            seq_length (array-like, optional): The length of each sequence in the batch. It should be a 1D tensor of shape (batch_size,) containing the sequence lengths. If not provided, it will be calculated
based on the maximum length of sequences in the batch.
        
        Returns:
            None: This method does not return any value directly. However, it performs computations to construct the CRF.
        
        Raises:
            ValueError: If the dimensions of emissions and tags are not compatible for the operations.
            TypeError: If the provided data types are not compatible with the expected types.
            RuntimeError: If an error occurs during the computation of the CRF, such as invalid input or inconsistent data.
        """
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
        r"""
        This method computes the score of a given sequence of emissions and tags using the Conditional Random Field (CRF) algorithm.
        
        Args:
            self (CRF): The CRF instance.
            emissions (Tensor): A 3D tensor containing the emission scores for each tag at each time step for all sequences in the batch. The shape of the tensor is (seq_length, batch_size, num_tags).
            tags (Tensor): A 2D tensor containing the predicted tags for each sequence in the batch. The shape of the tensor is (seq_length, batch_size).
            seq_ends (Tensor): A 1D tensor containing the indices of the ends of sequences in the batch.
            mask (Tensor): A 1D tensor indicating whether each time step in each sequence should be included in the score computation. The shape of the tensor is (seq_length,).
        
        Returns:
            None: This method does not return a value but updates the score attribute of the CRF instance.
        
        Raises:
            ValueError: If the shapes of the input tensors are incompatible or if any input tensor has an invalid shape.
            TypeError: If the data types of the input tensors are not compatible with the operations in the method.
        """
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
        r"""
        Compute the normalizer value for the Conditional Random Field (CRF).
        
        Args:
            self (CRF): An instance of the CRF class.
            emissions (Tensor): A tensor of shape (seq_length, num_tags) containing emission scores for each tag at each position in the input sequence.
            mask (Tensor): A binary tensor of shape (seq_length,) indicating the valid positions in the input sequence.
        
        Returns:
            None
        
        Raises:
            TypeError: If the input parameters are not of the correct type.
            ValueError: If the input tensors have incompatible shapes.
        
        This method computes the normalizer value for the CRF by iterating over the input sequence and calculating the scores for all possible tag sequences. The normalizer value is the sum of the
exponentiated scores.
        
        The emissions parameter represents the emission scores for each tag at each position in the input sequence. It should be a tensor of shape (seq_length, num_tags). The emission scores are used to
calculate the score for each tag sequence.
        
        The mask parameter is a binary tensor of shape (seq_length,) indicating the valid positions in the input sequence. Only the positions with a value of 1 in the mask are considered during the computation
of the normalizer.
        
        The method starts by initializing the score with the sum of the start transition scores and the emission scores for the first position in the input sequence. Then, it iterates over the remaining
positions in the sequence and updates the score using the transition scores and the emission scores for each position. The scores are calculated by expanding the score tensor and the emission tensor to the
appropriate dimensions and adding them together. The logsumexp function is then applied to obtain the updated score. The score is updated based on the mask, with only the valid positions being updated.
        
        Finally, the end transition scores are added to the score, and the logsumexp function is applied again to obtain the final normalizer value.
        
        Note: This method modifies the score and does not return any value.
        """
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
        r"""
        Performs Viterbi decoding on a given sequence of emissions and a mask.
        
        Args:
            self: An instance of the CRF class.
            emissions (ndarray): A 2-dimensional array of shape (sequence_length, num_classes) containing the emission scores for each class at each position in the sequence.
            mask (ndarray): A 1-dimensional array of shape (sequence_length,) representing the mask indicating valid positions in the sequence.
        
        Returns:
            score (ndarray): A 1-dimensional array of shape (num_classes,) containing the Viterbi score at the end of the sequence.
            history (tuple): A tuple of length (sequence_length - 1) containing the indices of the most likely classes at each position in the sequence.
        
        Raises:
            None.
        """
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
