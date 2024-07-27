# Copyright 2023 Huawei Technologies Co., Ltd
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

"""
Logits process
"""

import math
import inspect
import logging
from typing import List, Iterable, Union, Optional, Callable, Tuple, Dict
import mindspore
import numpy as np
from mindnlp.core import ops
from mindnlp.core.nn import functional as F


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""
    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""
    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """Torch method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """
    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor, **kwargs) -> mindspore.Tensor:
        """
        This method processes input_ids and scores using a list of processors.
        
        Args:
            self (LogitsProcessorList): The instance of the LogitsProcessorList class.
            input_ids (mindspore.Tensor): The input tensor containing the IDs of tokens to be processed.
            scores (mindspore.Tensor): The input tensor containing the scores to be processed.
        
        Returns:
            mindspore.Tensor: The processed scores tensor.
        
        Raises:
            ValueError: If not all the required parameters for a processor are passed to the logits processor.
        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class HammingDiversityLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces diverse beam search. Note that this logits processor is only effective for
    [`PreTrainedModel.group_beam_search`]. See [Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence
    Models](https://arxiv.org/pdf/1610.02424.pdf) for more details.

    Args:
        diversity_penalty (`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled.
        num_beams (`int`):
            Number of beams used for group beam search. See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more
            details.
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
    """
    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        """
        Initializes a new instance of the HammingDiversityLogitsProcessor class.
        
        Args:
            self: The instance of the class.
            diversity_penalty (float): The penalty factor for diversity. It should be a positive floating-point number.
            num_beams (int): The number of beams to use in the beam search. It should be an integer greater than 1.
            num_beam_groups (int): The number of beam groups. It should be an integer greater than 1 and less than or
                equal to num_beams.
        
        Returns:
            None.
        
        Raises:
            ValueError: If diversity_penalty is not a float or is not strictly larger than 0.
            ValueError: If num_beams is not an integer or is not strictly larger than 1.
            ValueError: If num_beam_groups is not an integer or is not strictly larger than 1.
            ValueError: If num_beam_groups is larger than num_beams.
        """
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError("`diversity_penalty` should be a float strictly larger than 0.")
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(
            self,
            input_ids: mindspore.Tensor,
            scores: mindspore.Tensor,
            current_tokens: mindspore.Tensor,
            beam_group_idx: int,
    ) -> mindspore.Tensor:
        """
        This method calculates the diversity penalty and updates the input scores based on the previous group tokens.
        
        Args:
            self: The instance of the HammingDiversityLogitsProcessor class.
            input_ids (mindspore.Tensor): The input tensor representing the tokenized input.
            scores (mindspore.Tensor): The tensor containing scores for each token.
            current_tokens (mindspore.Tensor): The tensor containing the current tokens.
            beam_group_idx (int): The index of the beam group.
        
        Returns:
            mindspore.Tensor: Returns the updated scores tensor after applying the diversity penalty.
        
        Raises:
            ValueError: If the input_ids, scores, current_tokens, or beam_group_idx are of incorrect or incompatible types.
            IndexError: If the beam_group_idx is out of range.
            RuntimeError: If there is an issue with the calculation or update process.
        """
        # hamming diversity: penalise using same token in current group which was used in previous groups at
        # the same time step
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        for batch_idx in range(batch_size):
            # predicted tokens of last time step of previous groups
            previous_group_tokens = current_tokens[
                                    batch_idx * self._num_beams: batch_idx * self._num_beams + group_start_idx
                                    ]
            token_frequency = ops.bincount(previous_group_tokens, minlength=vocab_size)
            scores[batch_idx * group_size: (batch_idx + 1) * group_size] -= self._diversity_penalty * token_frequency

        return scores


class EncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on tokens that are not in the original input.

    Args:
        hallucination_penalty (`float`):
            The parameter for hallucination penalty. 1.0 means no penalty.
        encoder_input_ids (`mindspore.Tensor`):
            The encoder_input_ids that should not be repeated within the decoder ids.
    """
    def __init__(self, penalty: float, encoder_input_ids: mindspore.Tensor):
        """
        Initializes an instance of the EncoderRepetitionPenaltyLogitsProcessor class.

        Args:
            self: The instance of the class.
            penalty (float): The penalty value for repetition. Must be a strictly positive float.
            encoder_input_ids (mindspore.Tensor): The input tensor of the encoder.

        Returns:
            None.

        Raises:
            ValueError: If `penalty` is not a strictly positive float.

        """
        if not isinstance(penalty, float) or (penalty <= 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = 1 / penalty
        self.encoder_input_ids = encoder_input_ids

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method calculates and applies repetition penalty to the logits based on the input scores.

        Args:
            self (EncoderRepetitionPenaltyLogitsProcessor): The instance of the EncoderRepetitionPenaltyLogitsProcessor
                class.
            input_ids (mindspore.Tensor): The input tensor containing the token ids.
            scores (mindspore.Tensor): The input tensor containing the original scores.

        Returns:
            mindspore.Tensor: Returns a tensor with repetition penalty applied to the logits.

        Raises:
            ValueError: If the dimensions of input_ids and scores do not match.
            TypeError: If the input_ids or scores are not instances of mindspore.Tensor.
            RuntimeError: If there is an issue with the scatter operation during processing.
        """
        score = ops.gather(scores, 1, self.encoder_input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = ops.where(score < 0, score * self.penalty, score / self.penalty)

        scores = ops.scatter(scores, 1, self.encoder_input_ids, score)
        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """
    def __init__(self, penalty: float):
        """
        Initializes a new RepetitionPenaltyLogitsProcessor with the specified penalty.

        Args:
            penalty (float): The penalty value to be applied to logits. It should be a strictly positive float.

        Returns:
            None.

        Raises:
            ValueError: If the penalty is not a float or if it is less than or equal to 0, a ValueError is raised.
        """
        if not isinstance(penalty, float) or (penalty <= 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method applies repetition penalty to the input logits based on the given input_ids and scores.

        Args:
            self (RepetitionPenaltyLogitsProcessor): An instance of the RepetitionPenaltyLogitsProcessor class.
            input_ids (mindspore.Tensor): A Tensor representing the input ids for which repetition penalty is applied.
            scores (mindspore.Tensor): A Tensor containing the scores used for applying repetition penalty.

        Returns:
            mindspore.Tensor: A Tensor with repetition penalty applied to the input logits.

        Raises:
            ValueError: If the input_ids and scores are not of the expected shape or type.
            IndexError: If there is an indexing error while processing the input_ids or scores.
            RuntimeError: If there is any runtime issue during the processing of the repetition penalty.

        Note:
            The repetition penalty factor is controlled by the 'penalty' attribute of the RepetitionPenaltyLogitsProcessor instance.
        """
        input_ids = ops.where(input_ids >= scores.shape[1], input_ids - scores.shape[1], input_ids)
        score = ops.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = ops.where(score < 0, score * self.penalty, score / self.penalty)

        scores = ops.scatter(scores, 1, input_ids, score)
        return scores


def _get_ngrams(ngram_size: int, prev_input_ids: mindspore.Tensor, num_hypos: int):
    """
    Args:
        ngram_size (int): The size of the n-grams to generate.
        prev_input_ids (mindspore.Tensor): The input tensor containing the previous tokens.
        num_hypos (int): The number of hypothesis.

    Returns:
        None.

    Raises:
        None
    """
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].asnumpy().tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Args:
        banned_ngrams (dict): A dictionary containing banned ngrams as keys and corresponding values.
        prev_input_ids (numpy.ndarray): An array of previous input ids.
        ngram_size (int): The size of the ngram to be generated.
        cur_len (int): The current length of the input sequence.

    Returns:
        list: A list of banned ngrams corresponding to the given ngram index, or an empty list if not found.

    Raises:
        None
    """
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(
        ngram_size: int, prev_input_ids: mindspore.Tensor, num_hypos: int, cur_len: int
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)

    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """
    def __init__(self, ngram_size: int):
        """
        Initializes a NoRepeatNGramLogitsProcessor object with the specified ngram size.

        Args:
            self: The instance of the class.
            ngram_size (int): The size of the n-gram to be used for processing the logits.
                It should be a strictly positive integer.

        Returns:
            None.

        Raises:
            ValueError: If the ngram_size is not an integer or is less than or equal to 0, a ValueError is raised
                with a descriptive error message.
        """
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method processes the logits for generating token sequences without repeating n-grams.

        Args:
            self (NoRepeatNGramLogitsProcessor): An instance of the NoRepeatNGramLogitsProcessor class.
            input_ids (mindspore.Tensor): A tensor containing input token IDs for the current batch.
                The shape of the tensor should be compatible with the model's input requirements.
            scores (mindspore.Tensor): A tensor containing the logits scores for each token in the vocabulary.
                The shape of the tensor should be compatible with the model's output logits.

        Returns:
            mindspore.Tensor: A tensor containing the modified logits scores after applying the no repeat n-gram processing.
                The modified scores ensure that tokens forming prohibited n-grams have their logits set to negative infinity.

        Raises:
            ValueError: If the shape of input_ids or scores is incompatible.
            TypeError: If input_ids or scores are not of type mindspore.Tensor.
        """
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores


class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces no repetition of encoder input ids n-grams for the decoder ids. See
    [ParlAI](https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_generator_agent.py#L1350).

    Args:
        encoder_ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur within the encoder input ids.
        encoder_input_ids (`int`):
            The encoder_input_ids that should not be repeated within the decoder ids.
    """
    def __init__(self, encoder_ngram_size: int, encoder_input_ids: mindspore.Tensor):
        """
        Initializes the EncoderNoRepeatNGramLogitsProcessor.

        Args:
            encoder_ngram_size (int): The size of the n-grams for encoding. Must be a strictly positive integer.
            encoder_input_ids (mindspore.Tensor): The input tensor for the encoder. If it has shape (N,),
                it will be unsqueezed to shape (1, N).

        Returns:
            None.

        Raises:
            ValueError: If `encoder_ngram_size` is not a strictly positive integer.
        """
        if not isinstance(encoder_ngram_size, int) or encoder_ngram_size <= 0:
            raise ValueError(
                f"`encoder_ngram_size` has to be a strictly positive integer, but is {encoder_ngram_size}"
            )
        self.ngram_size = encoder_ngram_size
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.batch_size = encoder_input_ids.shape[0]
        self.generated_ngrams = _get_ngrams(encoder_ngram_size, encoder_input_ids, self.batch_size)

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method processes logits to prevent generation of n-grams that have already appeared in the input sequence.

        Args:
            self (EncoderNoRepeatNGramLogitsProcessor): An instance of the EncoderNoRepeatNGramLogitsProcessor class.
            input_ids (mindspore.Tensor): A tensor containing the input token IDs.
                Shape: (batch_size, sequence_length)
            scores (mindspore.Tensor): A tensor containing the logits scores for each token.
                Shape: (num_hypos, vocab_size)

        Returns:
            mindspore.Tensor: A tensor containing the updated logits scores after processing.
                Shape: (num_hypos, vocab_size)

        Raises:
            TypeError: If the input_ids or scores parameters are not of type mindspore.Tensor.
            ValueError: If the dimensions of input_ids and scores do not match the expected shapes.
        """
        # B x num_beams
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = [
            _get_generated_ngrams(
                self.generated_ngrams[hypo_idx // num_beams], input_ids[hypo_idx], self.ngram_size, cur_len
            )
            for hypo_idx in range(num_hypos)
        ]

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores


class NoBadWordsLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be sampled.

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated. In order to get the token ids of the words
            that should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
            add_special_tokens=False).input_ids`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """
    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: Union[int, List[int]]):
        """
        This method initializes an instance of the NoBadWordsLogitsProcessor class.

        Args:
            self: The instance of the class.
            bad_words_ids (List[List[int]]): A list of lists containing the IDs of bad words. Each inner list represents
                a sequence of bad word IDs. The outer list contains multiple sequences of bad word IDs. The parameter
                is expected to be a non-empty list of lists of positive integers.
            eos_token_id (Union[int, List[int]]): An integer or a list of integers representing the end-of-sequence
                token ID(s). If a single integer is provided, it is converted to a list with a single element.
                If this parameter is None, it is automatically assigned an empty list. It is expected to be a positive
                integer or a list of positive integers.

        Returns:
            None.

        Raises:
            ValueError:
                - If `bad_words_ids` is not a non-empty list.
                - If `bad_words_ids` is not a list of lists.
                - If any list in `bad_words_ids` is not a list of positive integers.
                - If `eos_token_id` is not a positive integer or a list of positive integers.
                - If the banned words token sequences cannot have an empty list.
        """
        if not isinstance(bad_words_ids, List) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.")
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.")
        if any(
                any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
                for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}."
            )

        if eos_token_id is None:
            eos_token_id = []
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        bad_words_ids = list(
            filter(lambda bad_token_seq: all(bad_token_seq != [i] for i in eos_token_id), bad_words_ids)
        )
        self.bad_words_id_length_1 = []
        self.bad_words_id_length_greater_than_1 = []
        for word in bad_words_ids:
            if len(word) == 1:
                self.bad_words_id_length_1.append(word[0])
            else:
                self.bad_words_id_length_greater_than_1.append(word)

        self.static_bad_words_mask: Optional[mindspore.Tensor] = None

        for banned_token_seq in self.bad_words_id_length_greater_than_1:
            if len(banned_token_seq) == 0:
                raise ValueError(f"Banned words token sequences {bad_words_ids} cannot have an empty list")

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method is a part of the 'NoBadWordsLogitsProcessor' class and is called '__call__'.
        It processes the input tensors and applies the 'No Bad Words' logic to the scores.

        Args:
            self: An instance of the 'NoBadWordsLogitsProcessor' class.
            input_ids (mindspore.Tensor):
                A tensor containing the input IDs.

                - Type: mindspore.Tensor
                - Purpose: Holds the input IDs for processing.
            scores (mindspore.Tensor):
                A tensor containing the scores.

                - Type: mindspore.Tensor
                - Purpose: Represents the scores to be processed.

        Returns:
            mindspore.Tensor:
                A tensor containing the processed scores.

                - Type: mindspore.Tensor
                - Purpose: Represents the scores after applying the 'No Bad Words' logic.

        Raises:
            None.
        """
        if self.static_bad_words_mask is None and len(self.bad_words_id_length_1) > 0:
            self.static_bad_words_mask = self._calc_static_bad_word_mask(scores)

        dynamic_banned_tokens = self._calc_banned_bad_words_ids(input_ids.tolist())
        scores = self._set_scores_to_inf_for_banned_tokens(scores, dynamic_banned_tokens)

        return scores

    def _calc_static_bad_word_mask(self, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method calculates a static bad word mask based on the given scores in the NoBadWordsLogitsProcessor class.

        Args:
            self (NoBadWordsLogitsProcessor): The instance of the NoBadWordsLogitsProcessor class.
            scores (mindspore.Tensor): A tensor containing scores used to calculate the static bad word mask.

        Returns:
            mindspore.Tensor: A boolean tensor representing the static bad word mask. The mask is created by setting
                the value to 1 at specific indices defined by 'bad_words_id_length_1' in the 'scores' tensor.

        Raises:
            None
        """
        static_bad_words_mask = ops.zeros(scores.shape[1])
        static_bad_words_mask[self.bad_words_id_length_1] = 1
        return static_bad_words_mask.unsqueeze(0).bool()

    def _tokens_match(self, prev_tokens: List[int], tokens: List[int]) -> bool:
        """
        Method _tokens_match in the class NoBadWordsLogitsProcessor checks if a sequence of tokens matches the previous tokens.

        Args:
            self: Instance of the NoBadWordsLogitsProcessor class.
            prev_tokens (List[int]): List of integers representing previous tokens to compare against.
            tokens (List[int]): List of integers representing tokens to check for a match.

        Returns:
            bool: Returns True if the tokens match the previous tokens, otherwise False.

        Raises:
            None.
        """
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False
        return prev_tokens[-len(tokens):] == tokens

    def _calc_banned_bad_words_ids(self, prev_input_ids: List[List[int]]) -> Iterable[int]:
        """
        Calculates the banned bad word IDs based on the previous input IDs.

        Args:
            self (NoBadWordsLogitsProcessor): An instance of the NoBadWordsLogitsProcessor class.
            prev_input_ids (List[List[int]]): A list of lists representing the previous input IDs.
                Each inner list contains integers representing token IDs.

        Returns:
            Iterable[int]: An iterable containing the banned bad word IDs.

        Raises:
            None

        Description:
            This method takes the previous input IDs and calculates the banned bad word IDs based on the pre-defined
            bad word sequences. It iterates over each slice of the previous input IDs and checks if any of the bad word
            sequences match with the preceding tokens. If a match is found, the corresponding bad word ID is added to
            the list of banned tokens. The banned tokens for each slice are then appended to the final list of
            banned tokens.

        Note:
            - The bad word sequences are specified in the 'bad_words_id_length_greater_than_1' attribute of the
            NoBadWordsLogitsProcessor class.

        Example:
            ```python
            >>> processor = NoBadWordsLogitsProcessor()
            >>> prev_input_ids = [[1, 2, 3], [4, 5, 6]]
            >>> banned_tokens = processor._calc_banned_bad_words_ids(prev_input_ids)
            >>> # banned_tokens will contain the banned bad word IDs based on the previous input IDs.
            ```
        """
        banned_tokens = []
        for prev_input_ids_slice in prev_input_ids:
            banned_tokens_slice = []
            for banned_token_seq in self.bad_words_id_length_greater_than_1:
                if self._tokens_match(prev_input_ids_slice, banned_token_seq[:-1]):
                    banned_tokens_slice.append(banned_token_seq[-1])

            banned_tokens.append(banned_tokens_slice)

        return banned_tokens

    def _set_scores_to_inf_for_banned_tokens(
            self, scores: mindspore.Tensor, banned_tokens: List[List[int]]
    ) -> mindspore.Tensor:
        """
        Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
        list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
        """
        banned_mask_list = []
        for idx, batch_banned_tokens in enumerate(banned_tokens):
            for token in batch_banned_tokens:
                # Eliminates invalid bad word IDs that are over the vocabulary size.
                if token <= scores.shape[1]:
                    banned_mask_list.append([idx, token])
                else:
                    logging.error(
                        "An invalid bad word ID is defined: %d. This ID is not contained in the "
                        "vocabulary, and is therefore ignored.", token
                    )
        if not banned_mask_list and self.static_bad_words_mask is None:
            return scores

        if banned_mask_list:
            banned_mask = mindspore.Tensor(banned_mask_list)
            indices = ops.ones(len(banned_mask))
            # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
            # [ 0  1  1 ]
            # [ 0  0  0 ]
            # [ 1  0  0 ]

            banned_mask = (
                mindspore.COOTensor(banned_mask,
                                    indices, scores.shape)
                .to_dense()
                .bool()
            )

            if self.static_bad_words_mask is not None:
                banned_mask = ops.bitwise_or(banned_mask, self.static_bad_words_mask)
        else:
            banned_mask = self.static_bad_words_mask

        scores = scores.masked_fill(banned_mask, -float("inf"))
        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """
    def __init__(self, min_length: int, eos_token_id: Union[int, List[int]]):
        """
        Initializes an instance of the MinLengthLogitsProcessor class.

        Args:
            self (MinLengthLogitsProcessor): The current instance of the MinLengthLogitsProcessor class.
            min_length (int): The minimum length of the processed logits. It must be a positive integer.
            eos_token_id (Union[int, List[int]]): The end-of-sequence token ID or a list of end-of-sequence token IDs.
                If an integer is provided, it will be converted to a list. It must be a list of positive integers.

        Returns:
            None.

        Raises:
            ValueError: If `min_length` is not a positive integer.
            ValueError: If `eos_token_id` is not a list of positive integers.
        """
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if not all(isinstance(i, int) for i in eos_token_id) or any(i < 0 for i in eos_token_id):
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        The '__call__' method processes the scores of a given input and applies a minimum length constraint to the logits.
        It takes three parameters: self, input_ids, and scores.

        Args:
            self: An instance of the MinLengthLogitsProcessor class.
            input_ids (mindspore.Tensor): The input tensor representing the tokenized input sequence.
            scores (mindspore.Tensor): The tensor containing the logits scores.

        Returns:
            mindspore.Tensor: The processed scores tensor after applying the minimum length constraint.

        Raises:
            None.
        """
        vocab_tensor = ops.arange(scores.shape[-1])
        eos_token_id = mindspore.tensor(self.eos_token_id)
        eos_token_mask = vocab_tensor == eos_token_id
        scores_processed = scores.copy()
        if input_ids.shape[-1] < self.min_length:
            scores_processed = ops.where(eos_token_mask, -math.inf, scores)
        return scores_processed


class MinNewTokensLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.

    Args:
        prompt_length_to_skip (`int`):
            The input tokens length.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """
    def __init__(self, prompt_length_to_skip: int, min_new_tokens: int, eos_token_id: int):
        """
        __init__

        Args:
            prompt_length_to_skip (int): The length of prompt to skip for processing.
            min_new_tokens (int): The minimum number of new tokens to consider for processing.
            eos_token_id (int): The ID of the end-of-sequence token.

        Returns:
            None.

        Raises:
            ValueError: If prompt_length_to_skip, min_new_tokens, or eos_token_id is not a positive integer.
        """
        for arg_name, arg_value in [
            ("prompt_length_to_skip", prompt_length_to_skip),
            ("min_new_tokens", min_new_tokens),
            ("eos_token_id", eos_token_id),
        ]:
            if not isinstance(arg_value, int) or arg_value < 0:
                raise ValueError(f"`{arg_name}` has to be a positive integer, but is {arg_value}")

        self.prompt_length_to_skip = prompt_length_to_skip
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method '__call__' in the class 'MinNewTokensLengthLogitsProcessor' processes input_ids and scores to adjust
        scores based on the length of new tokens.

        Args:
            self (MinNewTokensLengthLogitsProcessor): The instance of the MinNewTokensLengthLogitsProcessor class.
            input_ids (mindspore.Tensor): The input tensor containing token IDs.
            scores (mindspore.Tensor): The input tensor containing the scores associated with each token.

        Returns:
            mindspore.Tensor: Returns the updated scores tensor after processing based on the input_ids and
                prompt_length_to_skip attribute.

        Raises:
            ValueError: If the new_tokens_length is calculated to be less than the min_new_tokens threshold.
        """
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        if new_tokens_length < self.min_new_tokens:
            scores[:, self.eos_token_id] = -float("inf")

        return scores


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.
    """
    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, mindspore.Tensor], List[int]], num_beams: int):
        """
        Initialize the PrefixConstrainedLogitsProcessor object.

        Args:
            self: The object instance.
            prefix_allowed_tokens_fn (Callable[[int, mindspore.Tensor], List[int]]):
                A function that defines the allowed tokens for a given prefix. It takes an integer representing the
                batch size and a Tensor as input, and returns a list of integers representing the allowed tokens.
            num_beams (int): The number of beams to use during processing.

        Returns:
            None.

        Raises:
            TypeError: If prefix_allowed_tokens_fn is not a callable object or if num_beams is not an integer.
            ValueError: If num_beams is less than or equal to zero.
        """
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        '''
        Method '__call__' in the class 'PrefixConstrainedLogitsProcessor'.

        This method takes 3 parameters:

        Args:
            self: The instance of the class.
            input_ids (mindspore.Tensor): The input tensor containing token IDs.
                It is used to identify the batch and beam ID.
            scores (mindspore.Tensor): The input tensor containing scores for each token.

        Returns:
            mindspore.Tensor: Returns the processed tensor with added mask values.

        Raises:
            None
        '''
        mask = ops.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                mask[batch_id * self._num_beams + beam_id, self._prefix_allowed_tokens_fn(batch_id, sent)] = 0

        return scores + mask


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """
    def __init__(self, bos_token_id: int):
        """
        Initializes a new instance of the ForcedBOSTokenLogitsProcessor class.

        Args:
            self: The object itself.
            bos_token_id (int): The token ID for the beginning of sentence (BOS) token.
                This ID is used to identify the BOS token in the input sequence.

        Returns:
            None.

        Raises:
            None.
        """
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method, '__call__', is a part of the 'ForcedBOSTokenLogitsProcessor' class.
        It takes three parameters: self, input_ids, and scores. The method returns a value of type 'mindspore.Tensor'.

        Args:
            self: The instance of the 'ForcedBOSTokenLogitsProcessor' class.
            input_ids (mindspore.Tensor): The input tensor containing the IDs of the tokens.
            scores (mindspore.Tensor): The tensor containing the scores for each token.

        Returns:
            mindspore.Tensor: The tensor containing the modified scores.

        Raises:
            None

        This method modifies the scores tensor by adjusting the scores based on the input IDs.
        If the length of the input_ids tensor is 1, the scores for all tokens except the 'bos_token_id' are set to
        negative infinity, and the score for the 'bos_token_id' is set to 0. The modified scores tensor is then returned.
        """
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.bos_token_id]] = -float("inf")
            scores[:, self.bos_token_id] = 0
        return scores


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`Union[int, List[int]]`):
            The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a
            list to set multiple *end-of-sequence* tokens.
    """
    def __init__(self, max_length: int, eos_token_id: Union[int, List[int]]):
        """
        Initializes a ForcedEOSTokenLogitsProcessor object with the specified parameters.

        Args:
            max_length (int): The maximum length for processing logits. Must be a positive integer.
            eos_token_id (Union[int, List[int]]): The end-of-sequence token ID(s) to be considered.
                If a single integer is provided, it will be converted to a list.

        Returns:
            None.

        Raises:
            None.
        """
        self.max_length = max_length
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        '''
        This method processes the logits by forcing end-of-sequence (EOS) tokens and returns the updated scores.

        Args:
            self (ForcedEOSTokenLogitsProcessor): The instance of the ForcedEOSTokenLogitsProcessor class.
            input_ids (mindspore.Tensor): The input tensor containing token IDs.
            scores (mindspore.Tensor): The tensor containing the scores/logits for each token.

        Returns:
            mindspore.Tensor: Returns the updated scores after processing.

        Raises:
            None.
        '''
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i not in self.eos_token_id]] = \
                float(np.finfo(mindspore.dtype_to_nptype(scores.dtype)).min)
            for i in self.eos_token_id:
                scores[:, i] = 0
        return scores


class InfNanRemoveLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using
    the logits processor should only be used if necessary since it can slow down the generation method. `max_length` is
    reached.
    """
    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method '__call__' in the class 'InfNanRemoveLogitsProcessor' processes input scores by replacing
        infinite and NaN values.

        Args:
            self: An instance of the InfNanRemoveLogitsProcessor class.
            input_ids (mindspore.Tensor): A tensor containing the input IDs.
            scores (mindspore.Tensor): A tensor containing the scores to be processed.
                Any NaN values in the scores will be replaced with 0.0, and any infinite values will be replaced with
                the maximum value for the data type.

        Returns:
            mindspore.Tensor: A tensor containing the processed scores after replacing NaN and infinite values.

        Raises:
            None.
        """
        # set all nan values to 0.0
        scores[ops.isnan(scores)] = 0.0

        # set all inf values to max possible value
        scores[scores == float("inf")] = float(np.finfo(mindspore.dtype_to_nptype(scores.dtype)).max)
        return scores


class ExponentialDecayLengthPenalty(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that exponentially increases the score of the eos_token_id after regulation_start has been
    reached.

    Args:
        exponential_decay_length_penalty (`tuple(int, float)`, *optional*):
            This tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty
            starts and `decay_factor` represents the factor of exponential decay
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        input_ids_seq_length (`int`):
            The length of the input sequence.
    """
    def __init__(
            self, exponential_decay_length_penalty: Tuple, eos_token_id: Union[int, List[int]],
            input_ids_seq_length: int
    ):
        """
        Initializes an instance of the ExponentialDecayLengthPenalty class with the specified parameters.

        Args:
            self: The instance of the class.
            exponential_decay_length_penalty (Tuple):
                A tuple containing two elements:

                - The start point for the exponential decay length penalty regulation.
                - The factor for the exponential decay length penalty regulation.
            eos_token_id (Union[int, List[int]]): The ID or list of IDs representing the end-of-sequence token(s).
            input_ids_seq_length (int): The length of the input sequence.

        Returns:
            None.

        Raises:
            TypeError: If the 'exponential_decay_length_penalty' parameter is not a tuple.
            ValueError: If the 'exponential_decay_length_penalty' tuple does not contain exactly two elements.
            ValueError: If the 'input_ids_seq_length' parameter is not an integer.
            ValueError: If the 'eos_token_id' parameter is not an integer or a list of integers.
        """
        self.regulation_start = exponential_decay_length_penalty[0] + input_ids_seq_length
        self.regulation_factor = exponential_decay_length_penalty[1]
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method calculates exponential decay length penalty for input scores based on the length of input_ids.

        Args:
            self (ExponentialDecayLengthPenalty): An instance of the ExponentialDecayLengthPenalty class.
            input_ids (mindspore.Tensor): A tensor containing input IDs.
                This tensor represents the input sequence for which the length penalty is to be applied.
            scores (mindspore.Tensor): A tensor containing scores to be adjusted based on the length of input_ids.
                The scores represent the probability distribution for each token in the input sequence.

        Returns:
            mindspore.Tensor: A tensor containing the adjusted scores after applying the exponential decay length penalty.
                The returned tensor has the same shape as the input 'scores'.

        Raises:
            ValueError: If the length of input_ids is not consistent with the shape of scores.
            TypeError: If input_ids or scores are not of type mindspore.Tensor.
        """
        cur_len = input_ids.shape[-1]
        if cur_len > self.regulation_start:
            for i in self.eos_token_id:
                scores[:, i] = scores[:, i] * pow(self.regulation_factor, cur_len - self.regulation_start)
        return scores


class SuppressTokensLogitsProcessor(LogitsProcessor):
    r"""
    This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf`
    so that they are not sampled.
    """
    def __init__(self, suppress_tokens):
        """
        Initializes an instance of the SuppressTokensLogitsProcessor class.

        Args:
            self (object): The instance of the class.
            suppress_tokens (iterable): A list of tokens to be suppressed.

        Returns:
            None.

        Raises:
            TypeError: If the suppress_tokens parameter is not an iterable.
        """
        self.suppress_tokens = list(suppress_tokens)

    def __call__(self, input_ids, scores):
        """
        The '__call__' method in the 'SuppressTokensLogitsProcessor' class modifies the 'scores' array by setting
        the values of specific tokens to negative infinity. It takes three parameters: 'self', 'input_ids', and 'scores'.
        The method does not return any value.

        Args:
            self (SuppressTokensLogitsProcessor): An instance of the 'SuppressTokensLogitsProcessor' class.
            input_ids (Tensor): A tensor containing the input token IDs.
            scores (Tensor): A tensor containing the scores for each token.

        Returns:
            None: The method modifies the 'scores' array in-place.

        Raises:
            None.
        """
        scores[:, self.suppress_tokens] = -float("inf")
        return scores


class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
    r"""
    [`SuppressTokensAtBeginLogitsProcessor`] supresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` at not
    sampled at the begining of the generation.
    """
    def __init__(self, begin_suppress_tokens, begin_index):
        """
        Initializes a new instance of the SuppressTokensAtBeginLogitsProcessor class.

        Args:
            self (SuppressTokensAtBeginLogitsProcessor): The current instance of the class.
            begin_suppress_tokens (list): A list of tokens to suppress at the beginning of the logits.
            begin_index (int): The index indicating the beginning of the logits.

        Returns:
            None.

        Raises:
            None.
        """
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        self.begin_index = begin_index

    def __call__(self, input_ids, scores):
        """
        This method __call__ is a part of the class SuppressTokensAtBeginLogitsProcessor and is used to process
        input_ids and scores by suppressing certain tokens at the beginning.

        Args:
            self (object): The instance of the class SuppressTokensAtBeginLogitsProcessor.
            input_ids (numpy.ndarray): The input token IDs, expected to be a 2D array.
            scores (numpy.ndarray): The input logits scores, expected to be a 2D array.

        Returns:
            None: This method directly modifies the 'scores' array in place.

        Raises:
            None.
        """
        if input_ids.shape[1] == self.begin_index:
            scores[:, self.begin_suppress_tokens] = -float("inf")

        return scores


class ForceTokensLogitsProcessor(LogitsProcessor):
    r"""This processor takes a list of pairs of integers which indicates a mapping from generation indices to token
    indices that will be forced before sampling. The processor will set their log probs to `inf` so that they are
    sampled at their corresponding index."""
    def __init__(self, force_token_map: List[List[int]]):
        """
        Initializes a new instance of the ForceTokensLogitsProcessor class.

        Args:
            self: The instance of the class.
            force_token_map (List[List[int]]):
                A list of lists containing integer values representing the force token map.

        Returns:
            None.

        Raises:
            None.
        """
        self.force_token_map = dict(force_token_map)

    def __call__(self, input_ids, scores):
        """
        This method modifies the scores of input tokens based on a predefined set of force tokens.

        Args:
            self (ForceTokensLogitsProcessor): An instance of the ForceTokensLogitsProcessor class.
            input_ids (torch.Tensor): A tensor containing the input token IDs with shape (batch_size, sequence_length).
            scores (torch.Tensor): A tensor containing the scores for each token with shape (batch_size, sequence_length).

        Returns:
            None

        Raises:
            None
        """
        generation_idx = input_ids.shape[-1]
        current_token = self.force_token_map.get(generation_idx, None)
        if current_token is not None:
            scores[:, :] = -float("inf")
            scores[:, current_token] = 0
        return scores


class LogitNormalization(LogitsProcessor, LogitsWarper):
    r"""
    [`LogitsWarper`] and [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.
    """
    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """

        Description:
            This class provides a method for logit normalization.

        Args:
            self (LogitNormalization): The instance of the LogitNormalization class.
            input_ids (mindspore.Tensor): The input tensor containing the IDs.
                This tensor is used as an input to calculate the log softmax.
            scores (mindspore.Tensor): The tensor containing the scores to be normalized.
                The scores are normalized using log softmax along the last dimension.

        Returns:
            mindspore.Tensor: Returns the normalized scores in the form of a Tensor.
                The normalized scores are obtained by applying log softmax to the input scores.

        Raises:
            None
        """
        scores = F.log_softmax(scores, dim=-1)
        return scores


class TemperatureLogitsWarper(LogitsWarper):
    r"""
    [`TemperatureLogitsWarper`] for temperature (exponential scaling output probability distribution).
    Args:
        temperature (:obj:`float`):
            The value used to module the logits distribution.
    """
    def __init__(self, temperature: float):
        """
        Initializes a TemperatureLogitsWarper object with the provided temperature.

        Args:
            self: The object instance itself.
            temperature (float): The temperature value to be set for the TemperatureLogitsWarper object.
                Must be a strictly positive float.

        Returns:
            None.

        Raises:
            ValueError: If the provided temperature is not a float or is not strictly greater than 0.
        """
        if not isinstance(temperature, float) or not temperature > 0:
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method adjusts the input 'scores' by dividing them by a temperature value and returns the adjusted scores.

        Args:
            self (TemperatureLogitsWarper): The instance of the TemperatureLogitsWarper class.
            input_ids (mindspore.Tensor): The input tensor containing the IDs of the input data.
            scores (mindspore.Tensor): The input tensor containing the scores to be adjusted.

        Returns:
            mindspore.Tensor: A tensor containing the adjusted scores after dividing them by the temperature value.

        Raises:
            None
        """
        scores = scores / self.temperature
        return scores


class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """
    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        """
        Initializes an instance of the TopPLogitsWarper class.

        Args:
            top_p (float): The value representing the top cumulative probability for truncation.
                Must be a float greater than 0 and less than 1.
            filter_value (float, optional): The filter value used for truncation. Defaults to negative infinity.
            min_tokens_to_keep (int, optional): The minimum number of tokens to keep after truncation.
                Must be a positive integer greater than or equal to 1.

        Returns:
            None.

        Raises:
            ValueError:
                - If top_p is not a float greater than 0 or less than 1.
                - If min_tokens_to_keep is not a positive integer.
        """
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or min_tokens_to_keep < 1:
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method '__call__' in the class 'TopPLogitsWarper' applies the Top-p sampling strategy to filter out
        low probability tokens from the input scores tensor.

        Args:
            self: An instance of the TopPLogitsWarper class.
            input_ids (mindspore.Tensor): The input tensor containing the token IDs.
            scores (mindspore.Tensor): The input tensor containing the model scores for each token.

        Returns:
            mindspore.Tensor: A tensor representing the filtered scores after applying the Top-p sampling strategy.

        Raises:
            ValueError: If the filter_value is set to negative infinity.
            TypeError: If the input tensors are not of type mindspore.Tensor.
            RuntimeError: If an error occurs during the sorting, softmax calculation, or masking of scores.
        """
        if self.filter_value == -float("Inf"):
            self.filter_value = float(np.finfo(mindspore.dtype_to_nptype(scores.dtype)).min)
        # scores = ops.select(ops.isneginf(scores), mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(scores.dtype)).min), scores)
        sorted_logits, sorted_indices = ops.sort(scores, descending=False)
        cumulative_probs = ops.cumsum(ops.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0
        sorted_indices_to_remove = sorted_indices_to_remove.astype(mindspore.int32)

        # scatter sorted tensors to original indexing
        indices_to_remove = ops.scatter(sorted_indices_to_remove, 1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove.astype(mindspore.bool_), self.filter_value)
        return scores


class TopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """
    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        """Initialize the TopKLogitsWarper object.

        Args:
            top_k (int): The number of top logits to keep. Must be a positive integer.
            filter_value (float, optional): The value used to filter logits. Defaults to negative infinity.
            min_tokens_to_keep (int, optional): The minimum number of tokens to keep. Defaults to 1.

        Returns:
            None.

        Raises:
            ValueError: If top_k is not a positive integer.
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method, named '__call__', is defined within the 'TopKLogitsWarper' class and is used to process
        input_ids and scores to obtain a mindspore.Tensor result.

        Args:
            self: The instance of the 'TopKLogitsWarper' class.
            input_ids (mindspore.Tensor): The input tensor containing the input IDs.
            scores (mindspore.Tensor): The tensor containing the scores.

        Returns:
            mindspore.Tensor: A tensor containing the processed scores after applying the top-k warping.

        Raises:
            ValueError: If the filter_value is not set to -float('Inf').
            TypeError: If the input_ids or scores are not of type mindspore.Tensor.
            RuntimeError: If an error occurs during the execution of the method.
        """
        if self.filter_value == -float("Inf"):
            self.filter_value = float(np.finfo(mindspore.dtype_to_nptype(scores.dtype)).min)
        top_k = min(self.top_k, scores.shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < ops.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

class TypicalLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs typical decoding. See [Typical Decoding for Natural Language
    Generation](https://arxiv.org/abs/2202.00666) for more information.

    Args:
        mass (`float`, *optional*, defaults to 0.9):
            Value of typical_p between 0 and 1 inclusive, defaults to 0.9.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        """
        Initializes an instance of TypicalLogitsWarper.

        Args:
            mass (float, optional): The mass parameter representing the typicality weight.
                It should be a float between 0 and 1 exclusive. Defaults to 0.9.
            filter_value (float, optional): The filter value for logits. Defaults to negative infinity.
            min_tokens_to_keep (int, optional): The minimum number of tokens to keep.
                Should be a positive integer greater than or equal to 1. Defaults to 1.

        Returns:
            None.

        Raises:
            ValueError:
                - If mass is not a float between 0 and 1 exclusive.
                - If min_tokens_to_keep is not a positive integer.
        """
        mass = float(mass)
        if mass <= 0 or mass >= 1:
            raise ValueError(f"`typical_p` has to be a float > 0 and < 1, but is {mass}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method applies a warping function to the input scores to filter out low-confidence tokens based on
        their probability distribution.

        Args:
            self (TypicalLogitsWarper): The instance of the TypicalLogitsWarper class.
            input_ids (mindspore.Tensor): The input tensor containing the token IDs.
            scores (mindspore.Tensor): The input tensor containing the logits.

        Returns:
            mindspore.Tensor: A tensor containing the warped scores after filtering out low-confidence tokens.

        Raises:
            ValueError: If the input_ids and scores have incompatible shapes or types.
            RuntimeError: If an error occurs during the warping process.
        """
        # calculate entropy
        normalized = F.log_softmax(scores, dim=-1)
        p = ops.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = ops.abs((-normalized) - ent)
        sorted_scores, sorted_indices = ops.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = ops.cumsum(ops.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).axis(dim=1)
        last_ind.clamp_(max=sorted_scores.shape[-1] - 1)
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = ops.scatter(sorted_indices_to_remove, 1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class EpsilonLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs epsilon-sampling, i.e. restricting to tokens with `prob >= epsilon`. Takes the
    largest min_tokens_to_keep tokens if no tokens satisfy this constraint. See [Truncation Sampling as Language Model
    Desmoothing](https://arxiv.org/abs/2210.15191) for more information.

    Args:
        epsilon (`float`):
            If set to > 0, only the most tokens with probabilities `epsilon` or higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
        ...
        >>> set_seed(0)
        >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        ...
        >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")
        ...
        >>> # With sampling, the output is unexpected -- sometimes too unexpected.
        >>> outputs = model.generate(**inputs, do_sample=True)
        >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2
        ...
        >>> # With epsilon sampling, the output gets restricted to high-probability tokens. Note that this is similar to
        >>> # Top P sampling, which restricts tokens based on their cumulative probability.
        >>> # Pro tip: The paper recomends using `epsilon_cutoff` values between 3e-4 and 9e-4
        >>> outputs = model.generate(**inputs, do_sample=True, epsilon_cutoff=0.1)
        >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
        ```
    """
    def __init__(self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        """
        Initializes an instance of the EpsilonLogitsWarper class.

        Args:
            self: The instance of the class.
            epsilon (float): The value used for epsilon cutoff.
                It should be a float greater than 0 and less than 1.
            filter_value (float, optional): The filter value for the warping operation.
                Defaults to negative infinity.
            min_tokens_to_keep (int, optional): The minimum number of tokens to keep.
                It should be a strictly positive integer.

        Returns:
            None.

        Raises:
            ValueError: If epsilon is not within the range (0, 1) or if min_tokens_to_keep is less than 1.
        """
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`epsilon_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        self.epsilon = epsilon
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        '''
        This method takes three parameters:
        Args:
            self (EpsilonLogitsWarper): The instance of the EpsilonLogitsWarper class.
            input_ids (mindspore.Tensor): The input tensor containing the IDs.
            scores (mindspore.Tensor): The input tensor containing the scores.

        Returns:
            mindspore.Tensor: A tensor containing the modified scores after applying the epsilon logits warping.

        Raises:
            None.
        '''
        # Determine which indices to remove
        probabilities = ops.softmax(scores, dim=-1)
        indices_to_remove = probabilities < self.epsilon

        # Keep the words with the 'min_tokens_to_keep'-highest probabilities
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (scores < ops.topk(scores, top_k)[0][..., -1, None])

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class EtaLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs eta-sampling, a technique to filter out tokens with probabilities below a dynamic
    cutoff value, `eta`, which is calculated based on a combination of the hyperparameter `epsilon` and the entropy of
    the token probabilities, i.e. `eta := min(epsilon, sqrt(epsilon * e^-entropy(probabilities)))`. Takes the largest
    min_tokens_to_keep tokens if no tokens satisfy this constraint. It addresses the issue of poor quality in long
    samples of text generated by neural language models leading to more coherent and fluent text. See [Truncation
    Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more information. Note: `do_sample`
    must be set to `True` for this `LogitsWarper` to work.


    Args:
        epsilon (`float`):
            A float value in the range (0, 1). Hyperparameter used to calculate the dynamic cutoff value, `eta`. The
            suggested values from the paper ranges from 3e-4 to 4e-3 depending on the size of the model.
        filter_value (`float`, *optional*, defaults to -inf):
            All values that are found to be below the dynamic cutoff value, `eta`, are set to this float value. This
            parameter is useful when logits need to be modified for very low probability tokens that should be excluded
            from generation entirely.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Specifies the minimum number of tokens that must be kept for generation, regardless of their probabilities.
            For example, if `min_tokens_to_keep` is set to 1, at least one token will always be kept for generation,
            even if all tokens have probabilities below the cutoff `eta`.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
        ...
        >>> set_seed(0)
        >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        ...
        >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")
        ...
        >>> # With sampling, the output is unexpected -- sometimes too unexpected.
        >>> outputs = model.generate(**inputs, do_sample=True)
        >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2
        ...
        >>> # With eta sampling, the output gets restricted to high-probability tokens. You can see it as a dynamic form of
        >>> # epsilon sampling that adapts its cutoff probability based on the entropy (high entropy = lower cutoff).
        >>> # Pro tip: The paper recomends using `eta_cutoff` values between 3e-4 to 4e-3
        >>> outputs = model.generate(**inputs, do_sample=True, eta_cutoff=0.1)
        >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
        ```
    """
    def __init__(self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        """Initialize a new instance of the EtaLogitsWarper class.

        Args:
            self: The object itself.
            epsilon (float): The value to be used as epsilon. It should be a float between 0 and 1.
            filter_value (float, optional): The value to be used for filtering. Defaults to -float('Inf').
            min_tokens_to_keep (int, optional): The minimum number of tokens to keep. Defaults to 1.

        Returns:
            None.

        Raises:
            ValueError: If epsilon is not a float between 0 and 1.
            ValueError: If min_tokens_to_keep is not a positive integer.
        """
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`eta_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        self.epsilon = mindspore.tensor(epsilon, mindspore.float32)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method '__call__' in the class 'EtaLogitsWarper' takes three parameters:

        Args:
            self: The instance of the class.
            input_ids (mindspore.Tensor): The input tensor containing the IDs.
            scores (mindspore.Tensor): The input tensor containing the scores.

        Returns:
            mindspore.Tensor: Returns a tensor after applying certain operations on the input scores.

        Raises:
            None
        """
        # Calculate the adaptive cutoff
        probabilities = scores.softmax(dim=-1)
        entropy = mindspore.nn.probability.distribution.Categorical().entropy(scores)
        eta = ops.min(self.epsilon, ops.sqrt(self.epsilon) * ops.exp(-entropy))[..., None]
        indices_to_remove = probabilities < eta

        # Keep the words with the 'min_tokens_to_keep'-highest probabilities
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (scores < ops.topk(scores, top_k)[0][..., -1, None])

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

class SequenceBiasLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that applies an additive bias on sequences. The bias is applied to the last token of a sequence
    when the next generated token can complete it. Consequently, to take the most of biasing sequences with more than
    one token, consider using beam methods (to gracefully work around partially completed sequences that have a
    negative bias) and applying the bias to their prefixes (to ensure the bias is applied earlier).

    <Tip>

    In order to get the token ids of the sequences that you want to bias, make sure to set `add_prefix_space=True` when
    initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The
    `add_prefix_space` argument is only supported for some slow tokenizers, as fast tokenizers' prefixing behaviours
    come from `pre tokenizers`. Read more [here](https://hf-mirror.com/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        sequence_bias (`Dict[Tuple[int], float]`):
            Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. If a sequence has a length of 1, its bias
            will always be applied. Otherwise, the bias will only be applied if the sequence in question is about to be
            completed (in the token selection step after this processor is applied).

    Example:
        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        ...
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")
        ...
        >>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4)
        >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
        The full name of Donald is Donald J. Trump Jr
        ...
        >>> # Now let's control generation through a bias. Please note that the tokenizer is initialized differently!
        >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
        ...
        ...
        >>> def get_tokens_as_tuple(word):
        ...     return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])
        ...
        ...
        >>> # If we add a negative bias without beam search, it may become "stuck" in a prefix without good continuations
        >>> sequence_bias = {get_tokens_as_tuple("Trump"): -10.0}
        >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, sequence_bias=sequence_bias)
        >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
        The full name of Donald is Donald J. Donald,
        >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
        >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
        The full name of Donald is Donald Rumsfeld,
        >>> # We can also add a positive bias to nudge the model towards specific tokens or continuations
        >>> sequence_bias = {get_tokens_as_tuple("Donald Duck"): 10.0}
        >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
        >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
        The full name of Donald is Donald Duck.
        ```
    """
    def __init__(self, sequence_bias: Dict[Tuple[int], float]):
        """
        Initializes an instance of the SequenceBiasLogitsProcessor class.

        Args:
            self: The object instance.
            sequence_bias (Dict[Tuple[int], float]): A dictionary containing the sequence bias values.
                The keys are tuples of integers representing the sequence positions, and the values are floats
                representing the bias for each position.

        Returns:
            None.

        Raises:
            None.
        """
        self.sequence_bias = sequence_bias
        self._validate_arguments()

        # Bias variables that will be populated on the first call (for retrocompatibility purposes, the vocabulary size
        # is infered in the first usage, which inhibits initializing here)
        self.length_1_bias = None
        self.prepared_bias_variables = False

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        __call__

        This method processes the input_ids and scores to apply sequence bias to the logits.

        Args:
            self (SequenceBiasLogitsProcessor): The instance of the SequenceBiasLogitsProcessor class.
            input_ids (mindspore.Tensor): The input tensor containing the tokenized input sequence.
            scores (mindspore.Tensor): The input tensor containing the scores/logits to be processed.

        Returns:
            mindspore.Tensor: The processed scores/logits after applying sequence bias.

        Raises:
            ValueError: If the input_ids or scores are not of type mindspore.Tensor.
            ValueError: If the input_ids and scores do not have compatible shapes for processing.
        """
        # 1 - Prepares the bias tensors. This is only needed the first time the logit processor is called.
        if not self.prepared_bias_variables:
            self._prepare_bias_variables(scores)

        # 2 - prepares an empty bias to add
        bias = ops.zeros_like(scores)

        # 3 - include the bias from length = 1
        bias += self.length_1_bias

        # 4 - include the bias from length > 1, after determining which biased sequences may be completed.
        for sequence_ids, sequence_bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:  # the sequence is of length 1, already applied
                continue
            if len(sequence_ids) > input_ids.shape[1]:  # the sequence is longer than the context, ignore
                continue
            prefix_length = len(sequence_ids) - 1
            last_token = sequence_ids[-1]
            matching_rows = ops.eq(
                input_ids[:, -prefix_length:],
                mindspore.tensor(sequence_ids[:-1], dtype=input_ids.dtype),
            ).prod(dim=1)
            bias[:, last_token] += ops.where(
                matching_rows.bool(),
                mindspore.tensor(sequence_bias),
                mindspore.tensor(0.0),
            )

        # 5 - apply the bias to the scores
        scores = scores + bias
        return scores

class AlternatingCodebooksLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing alternated generation between the two codebooks of [`Bark`]'s fine submodel.

    Args:
        input_start_len (`int`):
            The length of the initial input sequence.
        semantic_vocab_size (`int`):
            Vocabulary size of the semantic part, i.e number of tokens associated to the semantic vocabulary.
        codebook_size (`int`):
            Number of tokens associated to the codebook.
    """
    def __init__(self, input_start_len: int, semantic_vocab_size: int, codebook_size: int):
        """
        Initializes an instance of the AlternatingCodebooksLogitsProcessor class.

        Args:
            self: The instance of the class.
            input_start_len (int): The starting length of the input sequence.
                Must be a non-negative integer.
            semantic_vocab_size (int): The size of the semantic vocabulary.
            codebook_size (int): The size of the codebook.

        Returns:
            None.

        Raises:
            ValueError: If `input_start_len` is not an integer or if it is a negative value.

        """
        if not isinstance(input_start_len, int) or input_start_len < 0:
            raise ValueError(f"`input_starting_length` has to be a non-negative integer, but is {input_start_len}")

        self.input_start_len = input_start_len
        self.semantic_vocab_size = semantic_vocab_size
        self.codebook_size = codebook_size

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        The '__call__' method in the 'AlternatingCodebooksLogitsProcessor' class processes the input tensors to
        manipulate the scores based on alternating codebooks.

        Args:
            self: An instance of the 'AlternatingCodebooksLogitsProcessor' class.
            input_ids (mindspore.Tensor):
                A tensor containing the input IDs.

                - Shape: (batch_size, sequence_length).
                - The sequence length represents the length of the input IDs.
            scores (mindspore.Tensor):
                A tensor containing the scores.

                - Shape: (batch_size, vocabulary_size).
                - The vocabulary size represents the total number of elements in the vocabulary.

        Returns:
            mindspore.Tensor:
                A tensor representing the modified scores.

                - Shape: (batch_size, vocabulary_size).

        Raises:
            None.
        """
        curr_len = input_ids.shape[-1]

        # even -> first codebook, odd -> second codebook
        is_first_codebook = ((curr_len - self.input_start_len) % 2) == 0

        if is_first_codebook:
            scores[:, : self.semantic_vocab_size] = -float("inf")
            scores[:, self.semantic_vocab_size + self.codebook_size :] = -float("inf")
        else:
            scores[:, : self.semantic_vocab_size + self.codebook_size] = -float("inf")

        return scores

class UnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""Logits processor for Classifier-Free Guidance (CFG). The processors
    computes a weighted average across scores from prompt conditional and prompt unconditional (or negative) logits,
    parameterized by the `guidance_scale`. The unconditional scores are computed internally by prompting `model` with
    the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (`float`):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale != 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality. A value smaller than 1 has the opposite effect, while
            making the negative prompt provided with negative_prompt_ids (if any) act as a positive prompt.
        model (`PreTrainedModel`):
            The model computing the unconditional scores. Supposedly the same as the one computing the conditional
            scores. Both models must use the same tokenizer.
        unconditional_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary for the unconditional branch. If unset, will default to
            the last token of the prompt.
        unconditional_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention mask for unconditional_ids.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to cache key/values during the negative prompt forward pass.


    Example:
        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        ...
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
        >>> out = model.generate(inputs["input_ids"], guidance_scale=1.5)
        >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'
        >>> # with a negative prompt
        >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
        >>> out = model.generate(inputs["input_ids"], guidance_scale=2, negative_prompt_ids=neg_inputs["input_ids"])
        >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'
        >>> # with a positive prompt
        >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
        >>> out = model.generate(inputs["input_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
        >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        "Today, a dragon flew over Paris, France, and I'm very happy to be here. I"
        ```
    """
    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_ids: Optional[mindspore.Tensor] = None,
        unconditional_attention_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = True,
    ):
        """
        Initializes the UnbatchedClassifierFreeGuidanceLogitsProcessor.

        Args:
            guidance_scale (float): The scale factor for guidance logits.
            model: The model used for processing.
            unconditional_ids (Optional[mindspore.Tensor], optional): The input tensor for unconditional context.
                Default is None.
            unconditional_attention_mask (Optional[mindspore.Tensor], optional): The attention mask for unconditional
                context. Default is None.
            use_cache (Optional[bool], optional): Flag to indicate whether to use caching. Default is True.

        Returns:
            None.

        Raises:
            None.
        """
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def get_unconditional_logits(self, input_ids):
        """get_unconditional_logits"""
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = ops.ones_like(
                    self.unconditional_context["input_ids"], dtype=mindspore.int64
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = ops.cat(
                [
                    self.unconditional_context["attention_mask"],
                    ops.ones_like(input_ids[:, -1:], dtype=mindspore.int64),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = ops.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        """
        This method processes input_ids and scores to compute guidance logits in the
        UnbatchedClassifierFreeGuidanceLogitsProcessor class.

        Args:
            self (object): Instance of the UnbatchedClassifierFreeGuidanceLogitsProcessor class.
            input_ids (tensor): Input tensor containing the IDs of the input data.
            scores (tensor): Tensor of scores to be processed.

        Returns:
            None.

        Raises:
            None.
        """
        scores = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        logits = self.get_unconditional_logits(input_ids)

        unconditional_logits = F.log_softmax(logits[:, -1], dim=-1)
        out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return out

class WhisperTimeStampLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that modifies the logits for the generation of timestamps in the transcription. When the input
    tokens are at a specific threshold, the processor sets the scores to negative infinity. The processor makes sure
    that timestamp tokens appear in pairs, by masking out the logits that would break this pairing pattern. This is
    done to maintain the consistency and structure of generated timestamps. It also ensures that when the predicted
    probability of sampling any of the timestamp token is greater than any individual non-timestamp token, those
    non-timestamp logits are set to negative infinity. This is done to ensure the generation of timestamps over other
    potential tokens.

    See [the paper](https://arxiv.org/abs/2212.04356) for more information.

    Args:
        generate_config (`GenerateConfig`):
            The generate config used to generate the output. The following parameters are required:

            - eos_token_id (`int`, *optional*, defaults to 50257):
            The id of the *end-of-sequence* token.
            - no_timestamps_token_id (`int`, *optional*, defaults to 50363):
            The id of the `"<|notimestamps|>"` token.
            - max_initial_timestamp_index (`int`, *optional*, defaults to 1):
            Used to set the maximum value of the initial timestamp. This is used to prevent the model from
            predicting timestamps that are too far in the future.

    Example:
        ``` python
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration,GenerationConfig
        >>> from datasets import load_dataset
        ...
        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[3]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features
        ...
        >>> #Displaying timestamps
        >>> generated_ids = model.generate(inputs=input_features, return_timestamps=True)
        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        >>> print("Transcription:", transcription)
        Transcription: <|startoftranscript|><|0.00|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all,
        and can<|6.44|><|6.44|> discover in it but little of rocky Ithaca.<|9.44|><|endoftext|>
        >>> #No timestamps & change EOS:
        >>> #This allows the user to select a specific token to terminate the sequence on, in this case it's the word "can"(460)
        >>> model.generation_config.eos_token_id = 460
        >>> generated_ids = model.generate(inputs=input_features,return_timestamps=False)
        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> print("Transcription:", transcription)
        Transcription:  He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can
        ```
    """
    def __init__(self, generate_config):  # support for the kwargs
        """
        This method initializes an instance of the WhisperTimeStampLogitsProcessor class.

        Args:
            self (object): The instance of the class itself.
            generate_config (object): An object containing configuration settings for generating timestamps.
                This parameter is required for initializing the processor.
                It should contain the following attributes:

                - eos_token_id (int): The token ID for the end of sentence.
                - no_timestamps_token_id (int): The token ID for no timestamps.
                - forced_decoder_ids (list): A list of forced decoder IDs.
                - max_initial_timestamp_index (int): The maximum initial timestamp index.

        Returns:
            None.

        Raises:
            ValueError: If the forced_decoder_ids attribute is empty or not provided in the generate_config object.
            IndexError: If the forced_decoder_ids attribute does not contain the required elements.
            TypeError: If generate_config is not an object or if any of its attributes are not of the expected types.
        """
        self.eos_token_id = generate_config.eos_token_id
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1

        self.begin_index = len(generate_config.forced_decoder_ids) + 2
        if generate_config.forced_decoder_ids[-1][1] == self.no_timestamps_token_id:
            self.begin_index -= 1
        self.max_initial_timestamp_index = generate_config.max_initial_timestamp_index

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method '__call__' in the class 'WhisperTimeStampLogitsProcessor' processes input_ids and scores to
        manipulate timestamps in the logits.

        Args:
            self: Instance of the 'WhisperTimeStampLogitsProcessor' class.
            input_ids (mindspore.Tensor): A tensor containing input IDs.
                Shape: [batch_size, sequence_length]
            scores (mindspore.Tensor): A tensor containing scores corresponding to input IDs.
                Shape: [batch_size, num_classes]

        Returns:
            mindspore.Tensor: A tensor representing processed scores after manipulating timestamps.

        Raises:
            ValueError: If input_ids shape is invalid or if max_initial_timestamp_index is not None and condition is not met.
            IndexError: If attempting to access elements out of bounds.
            RuntimeError: If an unknown error occurs during the processing.
        """
        # suppress <|notimestamps|> which is handled by without_timestamps
        scores[:, self.no_timestamps_token_id] = -float("inf")

        if input_ids.shape[1] == self.begin_index - 1:
            scores[:, :] = -float("inf")
            scores[:, self.timestamp_begin] = 0
            return scores

        # timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
        for k in range(input_ids.shape[0]):
            seq = list(input_ids[k, self.begin_index :].tolist())
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    scores[k, self.timestamp_begin :] = -float("inf")
                else:  # cannot be normal text tokens
                    scores[k, : self.eos_token_id] = -float("inf")

            # apply the `max_initial_timestamp` option
            if input_ids.shape[1] == self.begin_index and self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                scores[:, last_allowed + 1 :] = -float("inf")

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(scores.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[k, self.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.timestamp_begin].max()
            if not ops.isnan(timestamp_logprob) and timestamp_logprob > max_text_token_logprob:
                scores[k, : self.timestamp_begin] = -float("inf")

        return scores

class BarkEosPrioritizerLogitsProcessor(LogitsProcessor):
    r"""This processor ensures that the EOS token is selected if its probability is greater than the `min_eos_p`.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [Bark](https://hf-mirror.com/docs/transformers/en/model_doc/bark). See the model documentation for examples.

    </Tip>

    Args:
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        min_eos_p (`float`, *optional*):
            Minimum end of speech threshold.
    """
    def __init__(self, eos_token_id: Union[int, List[int]], min_eos_p: float):
        """
        Initializes an instance of the BarkEosPrioritizerLogitsProcessor class.

        Args:
            self: The instance of the class.
            eos_token_id (Union[int, List[int]]): An integer or a list of integers representing the end-of-sequence token ID(s).
                If an integer is provided, it will be converted to a list with that integer as its only element.
            min_eos_p (float): The minimum value for the end-of-sequence probability. Must be a positive float.
                If not None, it should be greater than 0. Otherwise, a ValueError will be raised.

        Returns:
            None.

        Raises:
            ValueError: Raised if min_eos_p is not a positive float or if it is None.
        """
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id
        if min_eos_p is not None and min_eos_p <= 0:
            raise ValueError(f"`min_eos_p` has to be a positive float, but is {min_eos_p}")
        self.min_eos_p = min_eos_p

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method processes input logits for early stopping based on a minimum probability threshold for the
        end-of-sequence token.

        Args:
            self (BarkEosPrioritizerLogitsProcessor): An instance of the BarkEosPrioritizerLogitsProcessor class.
            input_ids (mindspore.Tensor): A Tensor containing input IDs.
            scores (mindspore.Tensor): A Tensor containing logits scores.

        Returns:
            mindspore.Tensor: Processed logits scores with early stopping applied based on the minimum end-of-sequence
                probability threshold.

        Raises:
            None
        """
        if self.min_eos_p:
            probs = ops.softmax(scores.float(), dim=-1)
            # create scores full of -inf except for the eos_token_id
            early_stop_scores = ops.ones_like(scores) * -float("inf")
            early_stop_scores[:, self.eos_token_id] = scores[:, self.eos_token_id]

            do_early_stop = probs[:, self.eos_token_id] > self.min_eos_p
            do_early_stop = ops.any(do_early_stop, dim=1, keepdim=True)
            scores = ops.where(do_early_stop, early_stop_scores, scores)

        return scores


class ClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for classifier free guidance (CFG). The scores are split over the batch dimension,
    where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
    correspond to the unconditional logits (predicted from an empty or 'null' prompt). The processor computes a
    weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.

    See [the paper](https://arxiv.org/abs/2306.05284) for more information.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [MusicGen](https://hf-mirror.com/docs/transformers/main/en/model_doc/musicgen)

    </Tip>

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.

    Example:
        ```python
        >>> from transformers import AutoProcessor, MusicgenForConditionalGeneration
        ...
        >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        ...
        >>> inputs = processor(
        ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
        ...     padding=True,
        ...     return_tensors="pt",
        ... )
        >>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
        ```
        """

    def __init__(self, guidance_scale):
        """
        Initializes a new instance of the ClassifierFreeGuidanceLogitsProcessor class.

        Args:
            self: The instance of the class.
            guidance_scale (int): The scale of guidance for the classifier-free guidance processor. Must be greater than 1.

        Returns:
            None.

        Raises:
            ValueError: If the guidance_scale is not greater than 1, a ValueError is raised indicating the requirement
                for guidance_scale > 1 to use the classifier-free guidance processor.
        """
        if guidance_scale > 1:
            self.guidance_scale = guidance_scale
        else:
            raise ValueError(
                "Require guidance scale >1 to use the classifier free guidance processor, got guidance scale "
                f"{guidance_scale}."
            )

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        Performs processing on logits to generate processed scores for a classifier with free guidance.

        Args:
            self (ClassifierFreeGuidanceLogitsProcessor): An instance of the ClassifierFreeGuidanceLogitsProcessor class.
            input_ids (mindspore.Tensor): A tensor containing input IDs for the classifier.
            scores (mindspore.Tensor): A tensor containing logits for the classifier.

        Returns:
            mindspore.Tensor: A tensor containing the processed scores for the classifier.

        Raises:
            ValueError: If the shape of the scores tensor does not meet the required conditions.

        The '__call__' method processes the logits to generate processed scores for a classifier with free guidance.
        It expects two parameters: 'input_ids' and 'scores'. The method returns a tensor of type 'mindspore.Tensor'
        which contains the processed scores.

        The 'input_ids' parameter is a tensor that holds the input IDs for the classifier. It is used to determine the
        batch size and shape of the scores tensor. There are no specific restrictions on this parameter.

        The 'scores' parameter is a tensor that holds the logits for the classifier. It is expected to have twice the
        batch size of the input IDs tensor, with the first half of the batches corresponding to the conditional inputs
        and the second half corresponding to the unconditional inputs. The shape of the scores tensor should be
        (2 * input_ids.shape[0], ...). The method raises a ValueError if the shape of the scores tensor does not meet
        this requirement.

        The method splits the scores tensor into two parts: 'cond_logits' and 'uncond_logits'. 'cond_logits' represents
        the logits for the conditional inputs, while 'uncond_logits' represents the logits for the unconditional inputs.
        These logits are then processed using the guidance scale specified in the instance of the
        ClassifierFreeGuidanceLogitsProcessor class. The final processed scores are obtained by adding 'uncond_logits'
        to the difference between 'cond_logits' and 'uncond_logits', multiplied by the guidance scale.

        Note:
            This method assumes that the 'split' function splits the tensor into two parts along the first dimension (dim=0).
        """
        # simple check to make sure we have compatible batch sizes between our
        # logits scores (cond + uncond) and input ids (cond only)
        if scores.shape[0] != 2 * input_ids.shape[0]:
            raise ValueError(
                f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
                f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
                f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
            )
        unguided_bsz = scores.shape[0] // 2
        cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
        scores_processed = uncond_logits + (cond_logits - uncond_logits) * self.guidance_scale
        return scores_processed
