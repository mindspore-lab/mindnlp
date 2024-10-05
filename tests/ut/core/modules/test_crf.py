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
import itertools
import math
import random

import pytest
import mindspore
from mindnlp.core import nn, ops

from mindnlp.core.modules import CRF

RANDOM_SEED = 1478754

random.seed(RANDOM_SEED)
mindspore.set_seed(RANDOM_SEED)
try:
    mindspore.manual_seed(RANDOM_SEED)
except:
    pass

def compute_score(crf, emission, tag):
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
    for emit, t in zip(emission, tag):
        score += emit[t]

    return score


def make_crf(num_tags=5, batch_first=False):
    return CRF(num_tags, batch_first=batch_first)


def make_emissions(crf, seq_length=3, batch_size=2):
    em = ops.randn(seq_length, batch_size, crf.num_tags)
    if crf.batch_first:
        em = em.swapaxes(0, 1)
    return em


def make_tags(crf, seq_length=3, batch_size=2):
    # shape: (seq_length, batch_size)
    ts = mindspore.tensor([[random.randrange(crf.num_tags)
                        for b in range(batch_size)]
                       for _ in range(seq_length)],
                      dtype=mindspore.int64)
    if crf.batch_first:
        ts = ts.swapaxes(0, 1)
    return ts


class TestInit:
    def test_minimal(self):
        num_tags = 10
        crf = CRF(num_tags)

        assert crf.num_tags == num_tags
        assert not crf.batch_first
        assert isinstance(crf.start_transitions, nn.Parameter)
        assert crf.start_transitions.shape == (num_tags, )
        assert isinstance(crf.end_transitions, nn.Parameter)
        assert crf.end_transitions.shape == (num_tags, )
        assert isinstance(crf.transitions, nn.Parameter)
        assert crf.transitions.shape == (num_tags, num_tags)
        assert repr(crf) == f'CRF(num_tags={num_tags})'

    def test_full(self):
        crf = CRF(10, batch_first=True)
        assert crf.batch_first

    def test_nonpositive_num_tags(self):
        with pytest.raises(ValueError) as excinfo:
            CRF(0)
        assert 'invalid number of tags: 0' in str(excinfo.value)


class TestForward:
    def test_works_with_mask(self):
        crf = make_crf()
        seq_length, batch_size = 3, 2

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf, seq_length, batch_size)
        # mask should have size of (seq_length, batch_size)
        mask = mindspore.tensor([[1, 1, 1], [1, 1, 0]], dtype=mindspore.bool_).swapaxes(0, 1)

        # shape: ()
        llh = crf(emissions, tags, mask=mask)

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        mask = mask.swapaxes(0, 1)

        # Compute log likelihood manually
        manual_llh = 0.
        for emission, tag, mask_ in zip(emissions, tags, mask):
            seq_len = mask_.sum()
            emission, tag = emission[:seq_len], tag[:seq_len]
            numerator = compute_score(crf, emission, tag)
            all_scores = [
                compute_score(crf, emission, t)
                for t in itertools.product(range(crf.num_tags), repeat=seq_len)
            ]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh += numerator - denominator

        print(llh, manual_llh)
        assert_close(llh, manual_llh)
        # llh.backward()  # ensure gradients can be computed

    def test_works_without_mask(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)

        llh_no_mask = crf(emissions, tags)
        # No mask means the mask is all ones
        llh_mask = crf(emissions, tags, mask=ops.ones_like(tags, dtype=mindspore.bool_))

        assert_close(llh_no_mask, llh_mask)

    def test_batched_loss(self):
        crf = make_crf()
        batch_size = 10

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, batch_size=batch_size)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf, batch_size=batch_size)

        llh = crf(emissions, tags)
        assert ops.is_tensor(llh)
        assert llh.shape == ()

        total_llh = 0.
        for i in range(batch_size):
            # shape: (seq_length, 1, num_tags)
            emissions_ = emissions[:, i, :].unsqueeze(1)
            # shape: (seq_length, 1)
            tags_ = tags[:, i].unsqueeze(1)
            # shape: ()
            total_llh += crf(emissions_, tags_)

        assert_close(llh, total_llh)

    def test_reduction_none(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)

        seq_length, batch_size = tags.shape

        llh = crf(emissions, tags, reduction='none')

        assert ops.is_tensor(llh)
        assert llh.shape == (batch_size, )

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.swapaxes(0, 1)

        # Compute log likelihood manually
        manual_llh = []
        for emission, tag in zip(emissions, tags):
            numerator = compute_score(crf, emission, tag)
            all_scores = [
                compute_score(crf, emission, t)
                for t in itertools.product(range(crf.num_tags), repeat=seq_length)
            ]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh.append((numerator - denominator).item())

        assert_close(llh, mindspore.tensor(manual_llh))

    def test_reduction_mean(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)

        seq_length, batch_size = tags.shape

        llh = crf(emissions, tags, reduction='mean')

        assert ops.is_tensor(llh)
        assert llh.shape == ()

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.swapaxes(0, 1)

        # Compute log likelihood manually
        manual_llh = 0
        for emission, tag in zip(emissions, tags):
            numerator = compute_score(crf, emission, tag)
            all_scores = [
                compute_score(crf, emission, t)
                for t in itertools.product(range(crf.num_tags), repeat=seq_length)
            ]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh += numerator - denominator

        assert_close(llh, manual_llh / batch_size)

    def test_reduction_token_mean(self):
        crf = make_crf()
        seq_length, batch_size = 3, 2

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf, seq_length, batch_size)
        # mask should have size of (seq_length, batch_size)
        mask = mindspore.tensor([[1, 1, 1], [1, 1, 0]], dtype=mindspore.bool_).swapaxes(0, 1)

        llh = crf(emissions, tags, mask=mask, reduction='token_mean')

        assert ops.is_tensor(llh)
        assert llh.shape == ()

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        mask = mask.swapaxes(0, 1)

        # Compute log likelihood manually
        manual_llh, n_tokens = 0, 0
        for emission, tag, mask_ in zip(emissions, tags, mask):
            seq_len = mask_.sum()
            emission, tag = emission[:seq_len], tag[:seq_len]
            numerator = compute_score(crf, emission, tag)
            all_scores = [
                compute_score(crf, emission, t)
                for t in itertools.product(range(crf.num_tags), repeat=seq_len)
            ]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh += numerator - denominator
            n_tokens += seq_len

        assert_close(llh, manual_llh / n_tokens)

    def test_batch_first(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)
        llh = crf(emissions, tags)

        crf_bf = make_crf(batch_first=True)
        # Copy parameter values from non-batch-first CRF; requires_grad must be False
        # to avoid runtime error of in-place operation on a leaf variable
        crf_bf.start_transitions.assign_value(crf.start_transitions)
        crf_bf.end_transitions.assign_value(crf.end_transitions)
        crf_bf.transitions.assign_value(crf.transitions)

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.swapaxes(0, 1)
        llh_bf = crf_bf(emissions, tags)

        assert_close(llh, llh_bf)

    def test_emissions_has_bad_number_of_dimension(self):
        emissions = ops.randn(1, 2)
        tags = ops.ones(2, 2, dtype=mindspore.int64)
        crf = make_crf()

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags)
        assert 'emissions must have dimension of 3, got 2' in str(excinfo.value)

    def test_emissions_and_tags_size_mismatch(self):
        emissions = ops.randn(1, 2, 3)
        tags = ops.ones(2, 2, dtype=mindspore.int64)
        crf = make_crf(3)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags)
        assert (
            'the first two dimensions of emissions and tags must match, '
            'got (1, 2) and (2, 2)') in str(excinfo.value)

    def test_emissions_last_dimension_not_equal_to_number_of_tags(self):
        emissions = ops.randn(1, 2, 3)
        tags = ops.ones(1, 2, dtype=mindspore.int64)
        crf = make_crf(10)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags)
        assert 'expected last dimension of emissions is 10, got 3' in str(excinfo.value)

    def test_first_timestep_mask_is_not_all_on(self):
        emissions = ops.randn(3, 2, 4)
        tags = ops.ones(3, 2, dtype=mindspore.int64)
        mask = mindspore.tensor([[1, 1, 1], [0, 0, 0]], dtype=mindspore.bool_).swapaxes(0, 1)
        crf = make_crf(4)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags, mask=mask)
        assert 'mask of the first timestep must all be on' in str(excinfo.value)

        emissions = emissions.swapaxes(0, 1)
        tags = tags.swapaxes(0, 1)
        mask = mask.swapaxes(0, 1)
        crf = make_crf(4, batch_first=True)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags, mask=mask)
        assert 'mask of the first timestep must all be on' in str(excinfo.value)

    def test_invalid_reduction(self):
        crf = make_crf()
        emissions = make_emissions(crf)
        tags = make_tags(crf)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags, reduction='foo')
        assert 'invalid reduction: foo' in str(excinfo.value)


class TestDecode:
    def test_works_with_mask(self):
        crf = make_crf()
        seq_length, batch_size = 3, 2

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # mask should be (seq_length, batch_size)
        mask = mindspore.tensor([[1, 1, 1], [1, 1, 0]], dtype=mindspore.bool_).swapaxes(0, 1)

        best_tags = crf.decode(emissions, mask=mask)

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.swapaxes(0, 1)
        # shape: (batch_size, seq_length)
        mask = mask.swapaxes(0, 1)

        # Compute best tag manually
        for emission, best_tag, mask_ in zip(emissions, best_tags, mask):
            seq_len = mask_.sum()
            assert len(best_tag) == seq_len
            assert all(isinstance(t, int) for t in best_tag)
            emission = emission[:seq_len]
            manual_best_tag = max(
                itertools.product(range(crf.num_tags), repeat=seq_len),
                key=lambda t: compute_score(crf, emission, t))
            assert tuple(best_tag) == manual_best_tag

    def test_works_without_mask(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)

        best_tags_no_mask = crf.decode(emissions)
        # No mask means mask is all ones
        best_tags_mask = crf.decode(
            emissions, mask=emissions.new_ones(emissions.shape[:2], dtype=mindspore.bool_))

        assert best_tags_no_mask == best_tags_mask

    def test_batched_decode(self):
        crf = make_crf()
        batch_size, seq_length = 2, 3

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # shape: (seq_length, batch_size)
        mask = mindspore.tensor([[1, 1, 1], [1, 1, 0]], dtype=mindspore.bool_).swapaxes(0, 1)

        batched = crf.decode(emissions, mask=mask)

        non_batched = []
        for i in range(batch_size):
            # shape: (seq_length, 1, num_tags)
            emissions_ = emissions[:, i, :].unsqueeze(1)
            # shape: (seq_length, 1)
            mask_ = mask[:, i].unsqueeze(1)

            result = crf.decode(emissions_, mask=mask_)
            assert len(result) == 1
            non_batched.append(result[0])

        assert non_batched == batched

    def test_batch_first(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        best_tags = crf.decode(emissions)

        crf_bf = make_crf(batch_first=True)
        # Copy parameter values from non-batch-first CRF; requires_grad must be False
        # to avoid runtime error of in-place operation on a leaf variable
        crf_bf.start_transitions.assign_value(crf.start_transitions)
        crf_bf.end_transitions.assign_value(crf.end_transitions)
        crf_bf.transitions.assign_value(crf.transitions)

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.swapaxes(0, 1)
        best_tags_bf = crf_bf.decode(emissions)

        assert best_tags == best_tags_bf

    def test_emissions_has_bad_number_of_dimension(self):
        emissions = ops.randn(1, 2)
        crf = make_crf()

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions)
        assert 'emissions must have dimension of 3, got 2' in str(excinfo.value)

    def test_emissions_last_dimension_not_equal_to_number_of_tags(self):
        emissions = ops.randn(1, 2, 3)
        crf = make_crf(10)

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions)
        assert 'expected last dimension of emissions is 10, got 3' in str(excinfo.value)

    def test_emissions_and_mask_size_mismatch(self):
        emissions = ops.randn(1, 2, 3)
        mask = mindspore.tensor([[1, 1], [1, 0]], dtype=mindspore.bool_)
        crf = make_crf(3)

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions, mask=mask)
        assert (
            'the first two dimensions of emissions and mask must match, '
            'got (1, 2) and (2, 2)') in str(excinfo.value)

    def test_first_timestep_mask_is_not_all_on(self):
        emissions = ops.randn(3, 2, 4)
        mask = mindspore.tensor([[1, 1, 1], [0, 0, 0]], dtype=mindspore.bool_).swapaxes(0, 1)
        crf = make_crf(4)

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions, mask=mask)
        assert 'mask of the first timestep must all be on' in str(excinfo.value)

        emissions = emissions.swapaxes(0, 1)
        mask = mask.swapaxes(0, 1)
        crf = make_crf(4, batch_first=True)

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions, mask=mask)
        assert 'mask of the first timestep must all be on' in str(excinfo.value)


def assert_close(actual, expected):
    assert ops.allclose(actual, expected, atol=1e-12, rtol=1e-6)
