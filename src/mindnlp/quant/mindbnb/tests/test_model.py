# Copyright 2024 Huawei Technologies Co., Ltd
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
import sys
import os
import mindspore

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integrations.quantization_bnb_8bit import quant_8bit
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer

mindspore.set_context(device_target="GPU")

tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
model = AutoModelForCausalLM.from_pretrained("Rocketknight1/falcon-rw-1b")
model.set_train(False)
# pdb.set_trace()
# for name, param in model.parameters_and_names():
#     print(name)
#     print(param)
#     print(param.data.asnumpy())
# quantization
quant_8bit(model)
# for name, param in model.parameters_and_names():
#     print(name)
#     print(param)
#     print(param.data.asnumpy())

# pdb.set_trace()
inputs = tokenizer("My favorite food is", return_tensors="ms")
# pdb.set_trace()
output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=10)
output_str = tokenizer.batch_decode(output_ids)[0]
print(output_ids)
print(output_str)
