# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test lr schedule """

from mindtext.classification.utils import polynomial_decay_scheduler


def test_lr_schedule():
    lrs = polynomial_decay_scheduler(lr=1, min_lr=0.0, decay_steps=1, total_update_num=3, warmup_steps=0, power=1.0)
    lrs = lrs.tolist()
    assert lrs == [1.0, 0, 0]
