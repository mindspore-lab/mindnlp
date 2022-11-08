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

import itertools
import math
import random
import pytest
import mindspore
import numpy as np
from mindspore import ops
from mindnlp.modules import CRF

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
        score += crf.transitions[cur_tag, next_tag]

    # Add emission score
    for emit, tag_i in zip(emission, tag):
        score += emit[tag_i]

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


class TestForward:
    """
    Test forward
    """

    def test_works_without_mask(self):
        """
        Test works without mask
        """
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)

        llh_no_mask = crf(emissions, tags)
        # No mask means the mask is all ones
        llh_mask = crf(emissions, tags, seq_length=ops.fill(mindspore.int64, (tags.shape[1],), tags.shape[0]))

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
