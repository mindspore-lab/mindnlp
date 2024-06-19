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
"""
Test CRF
"""
# pylint: disable=W0640
# pylint: disable=W0212

import itertools
import math
import random
import pytest
import mindspore
import numpy as np
from ddt import ddt, data
from mindspore import ops, Tensor
from mindnlp import ms_jit
from mindnlp.modules import CRF
from mindnlp.modules.crf import sequence_mask
from ...common import MindNLPTestCase

RANDOM_SEED = 0

random.seed(RANDOM_SEED)
mindspore.set_seed(RANDOM_SEED)

def compute_score(crf, emission, tag):
    """
    Compute the score of a given tag sequence
    """
    # emission: (seq_length, num_tags)
    assert emission.ndim == 2
    assert emission.shape[0] == len(tag)
    assert emission.shape[1] == crf.num_tags
    assert all(0 <= t < crf.num_tags for t in tag)

    # Add transitions score
    score = crf.start_transitions[tag[0]] + crf.end_transitions[tag[-1]]
    for cur_tag, next_tag in zip(tag, tag[1:]):
        indices = ops.stack([Tensor(cur_tag), Tensor(next_tag)])
        score += ops.gather_nd(crf.transitions, indices.T)

    # Add emission score
    for i in range(emission.shape[0]):
        score += emission[i][tag[i]]

    return score


def make_crf(num_tags=5, batch_first=False, reduction='sum'):
    """
    Make CRF
    """
    return CRF(num_tags, batch_first=batch_first, reduction=reduction)


def make_emissions(crf, seq_length=3, batch_size=2):
    """
    Make emissions
    """
    emission = mindspore.Tensor(np.random.randn(seq_length, batch_size, crf.num_tags),
                                mindspore.float32)
    if crf.batch_first:
        emission = emission.swapaxes(0, 1)
    return emission


def make_tags(crf, seq_length=3, batch_size=2):
    """
    Make tags
    """
    # shape: (seq_length, batch_size)
    tags = mindspore.Tensor([[random.randrange(crf.num_tags)
                            for b in range(batch_size)]
                            for _ in range(seq_length)],
                          dtype=mindspore.int64)
    if crf.batch_first:
        tags = tags.swapaxes(0, 1)
    return tags


class TestInit:
    """Init Test"""
    def test_minimal(self):
        """
        Test minimal
        """
        num_tags = 10
        crf = CRF(num_tags)

        assert crf.num_tags == num_tags
        assert not crf.batch_first
        assert isinstance(crf.start_transitions, mindspore.Parameter)
        assert crf.start_transitions.shape == (num_tags, )
        assert isinstance(crf.end_transitions, mindspore.Parameter)
        assert crf.end_transitions.shape == (num_tags, )
        assert isinstance(crf.transitions, mindspore.Parameter)
        assert crf.transitions.shape == (num_tags, num_tags)
        assert repr(crf) == f'CRF(num_tags={num_tags})'

    def test_full(self):
        """
        Test full
        """
        crf = CRF(10, batch_first=True)
        assert crf.batch_first

    def test_nonpositive_num_tags(self):
        """
        Test nonpositive num_tags
        """
        with pytest.raises(ValueError) as excinfo:
            CRF(0)
        assert 'invalid number of tags: 0' in str(excinfo.value)

@ddt
class TestForward(MindNLPTestCase):
    """
    Test forward
    """

    @data(True, False)
    def test_works_without_mask(self, jit):
        """
        Test works without mask
        """
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)

        def forward(emissions, tags):
            llh_no_mask = crf(emissions, tags)
            # No mask means the mask is all ones
            llh_mask = crf(emissions, tags, seq_length=ops.fill(mindspore.int64, (tags.shape[1],), tags.shape[0]))
            return llh_no_mask, llh_mask

        if jit:
            forward = ms_jit(forward)

        llh_no_mask, llh_mask = forward(emissions, tags)

        assert np.allclose(llh_no_mask.asnumpy(), llh_mask.asnumpy(),rtol=1e-3,atol=1e-3)

    def test_batched_loss(self):
        """
        Test batched loss
        """
        crf = make_crf()
        batch_size = 10

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, batch_size=batch_size)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf, batch_size=batch_size)

        llh = crf(emissions, tags)
        assert llh.shape == ()

        total_llh = 0.
        for i in range(batch_size):
            # shape: (seq_length, 1, num_tags)
            emissions_ = emissions[:, i, :].expand_dims(1)
            # shape: (seq_length, 1)
            tags_ = tags[:, i].expand_dims(1)
            # shape: ()
            total_llh += crf(emissions_, tags_)

        assert np.allclose(llh.asnumpy(), total_llh.asnumpy(),rtol=1e-3,atol=1e-3)

    @pytest.mark.skip('transpose operator error on macOS and Windows')
    def test_reduction_none(self):
        """
        Test reduction none
        """
        crf = make_crf(reduction='none')
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)

        seq_length, batch_size = tags.shape

        llh = crf(emissions, tags)

        assert llh.shape == (batch_size, )

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.swapaxes(0, 1)

        # Compute log likelihood manually
        manual_llh = []
        for emission, tag in zip(emissions, tags):
            numerator = compute_score(crf, emission, tag).asnumpy()
            all_scores = [
                compute_score(crf, emission, t).asnumpy()
                for t in itertools.product(range(crf.num_tags), repeat=seq_length)
            ]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh.append(denominator - numerator)

        for llh_, manual_llh_ in zip(llh, manual_llh):
            assert np.allclose(llh_.asnumpy(), manual_llh_,rtol=1e-3,atol=1e-3)

    @pytest.mark.skip('transpose operator error on macOS and Windows')
    def test_reduction_mean(self):
        """
        Test reduction mean
        """
        crf = make_crf(reduction='mean')
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)

        seq_length, batch_size = tags.shape

        llh = crf(emissions, tags)

        assert llh.shape == ()

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.swapaxes(0, 1)

        # Compute log likelihood manually
        manual_llh = 0
        for emission, tag in zip(emissions, tags):
            numerator = compute_score(crf, emission, tag).asnumpy()
            all_scores = [
                compute_score(crf, emission, t).asnumpy()
                for t in itertools.product(range(crf.num_tags), repeat=seq_length)
            ]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh += denominator - numerator

        assert np.allclose(llh.asnumpy(), manual_llh / batch_size, rtol=1e-3, atol=1e-3)

    def test_batch_first(self):
        """
        Test batch first
        """
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)
        llh = crf(emissions, tags)

        crf_bf = make_crf(batch_first=True)
        # Copy parameter values from non-batch-first CRF; requires_grad must be False
        # to avoid runtime error of in-place operation on a leaf variable
        crf_bf.start_transitions.set_data(crf.start_transitions)
        crf_bf.end_transitions.set_data(crf.end_transitions)
        crf_bf.transitions.set_data(crf.transitions)
        crf_bf.start_transitions.requires_grad = False
        crf_bf.end_transitions.requires_grad = False
        crf_bf.transitions.requires_grad = False

        # shape: (batch_size, seq_length, num_tags)
        emissions_bf = emissions.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        tags_bf = tags.swapaxes(0, 1)
        llh_bf = crf_bf(emissions_bf, tags_bf)

        assert np.allclose(llh.asnumpy(), llh_bf.asnumpy(), rtol=1e-3, atol=1e-3)

@ddt
class TestDecode(MindNLPTestCase):
    """test crf decoding."""
    @data(True, False)
    def test_works_with_mask(self, jit):
        """test works with mask."""
        crf = make_crf()
        seq_length, batch_size = 3, 2
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # seq_length should be (batch_size,)
        seq_length = mindspore.Tensor([3, 2], mindspore.int64)

        def forward(emissions, seq_length):
            score, history = crf(emissions, seq_length=seq_length)
            return score, history

        if jit:
            forward = ms_jit(forward)

        score, history = forward(emissions, seq_length)
        best_tags = crf.post_decode(score, history, seq_length)
        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        mask = sequence_mask(seq_length, emissions.shape[1], True)
        # Compute best tag manually
        for emission, best_tag, mask_ in zip(emissions, best_tags, mask):
            seq_len = mask_.sum().astype(mindspore.int64)
            assert len(best_tag) == seq_len
            # assert all(isinstance(t, int) for t in best_tag)
            emission = emission[:len(best_tag)]
            manual_best_tag = max(
                itertools.product(range(crf.num_tags), repeat=seq_len),
                key=lambda t: compute_score(crf, emission, t))

            assert tuple(best_tag) == manual_best_tag

    def test_works_without_mask(self):
        """test works without mask."""
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)

        best_tags_no_mask = crf(emissions)
        # No mask means mask is all ones
        best_tags_mask = crf._viterbi_decode(
            emissions, mask=ops.ones(emissions.shape[:2], dtype=mindspore.bool_))

        assert (best_tags_no_mask[0] == best_tags_mask[0]).all()

    def test_batched_decode(self):
        """test batched decode."""
        crf = make_crf()
        batch_size, seq_length = 2, 3

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # shape: (seq_length, batch_size)
        batched = crf(emissions)[0]

        non_batched = []
        for i in range(batch_size):
            # shape: (seq_length, 1, num_tags)
            emissions_ = emissions[:, i, :].expand_dims(1)
            # shape: (seq_length, 1)
            result = crf(emissions_)
            assert len(result) == 2
            non_batched.append(result[0])

        non_batched = ops.concat(non_batched)
        assert (non_batched == batched).all()

    def test_batch_first(self):
        """test batch first."""
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        best_tags = crf(emissions)

        crf_bf = make_crf(batch_first=True)
        # Copy parameter values from non-batch-first CRF; requires_grad must be False
        # to avoid runtime error of in-place operation on a leaf variable
        crf_bf.start_transitions.set_data(crf.start_transitions)
        crf_bf.end_transitions.set_data(crf.end_transitions)
        crf_bf.transitions.set_data(crf.transitions)

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.swapaxes(0, 1)
        best_tags_bf = crf_bf(emissions)

        assert (best_tags[0] == best_tags_bf[0]).all()

    # def test_emissions_has_bad_number_of_dimension(self):
    #     """test emission has bad number of dimension."""
    #     emissions = Tensor(np.random.randn(1, 2), mindspore.float32)
    #     crf = make_crf()

    #     with pytest.raises(Exception) as excinfo:
    #         crf(emissions)
    #         print(excinfo)

    def test_emissions_last_dimension_not_equal_to_number_of_tags(self):
        """test emission last dimension not equal to number of tags."""
        emissions = Tensor(np.random.randn(1, 2, 3), mindspore.float32)
        crf = make_crf(10)

        with pytest.raises(Exception) as excinfo:
            crf(emissions)
            print(excinfo)
