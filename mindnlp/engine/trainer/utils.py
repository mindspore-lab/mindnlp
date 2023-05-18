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
utils for trainer.
"""

from mindspore.ops import value_and_grad

from mindnlp import ms_jit
from mindnlp._legacy.amp import all_finite

def get_default_forward_fn_with_loss_fn(network, loss_fn, loss_scaler):
    """get default forward function with loss function"""
    # forward function
    def forward_fn(inputs, labels):
        logits_list = ()
        logits = network(*inputs)
        if isinstance(logits, tuple):
            logits_list += logits
        else:
            logits_list += (logits,)

        loss = loss_fn(logits_list[0], *labels)
        loss = loss_scaler.scale(loss)
        return loss

    return forward_fn

def get_default_forward_fn_without_loss_fn(network, loss_scaler):
    """get default forward function without loss function"""
    def forward_fn(inputs):
        outputs_list = ()
        outputs = network(*inputs)
        if isinstance(outputs, tuple):
            outputs_list += outputs
        else:
            outputs_list += (outputs,)

        loss = loss_scaler.scale(outputs_list[0])
        return loss

    return forward_fn

def get_default_train_step_fn(forward_fn, optimizer, loss_scaler, check_gradients, jit, for_object_net=False):
    """get default train function"""
    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    def default_run_step(inputs, labels):
        """Core process of each step, including the forward propagation process and back propagation of data."""
        loss, grads = grad_fn(inputs, labels)
        loss = loss_scaler.unscale(loss)
        if check_gradients:
            status = all_finite(grads)
            if status:
                grads = loss_scaler.unscale(grads)
                optimizer(grads)
        else:
            optimizer(grads)
        return loss

    def default_run_step_for_obj_net(inputs):
        """Core process of each step, including the forward propagation process and back propagation of data."""
        loss, grads = grad_fn(inputs)
        loss = loss_scaler.unscale(loss)
        if check_gradients:
            status = all_finite(grads)
            if status:
                grads = loss_scaler.unscale(grads)
                optimizer(grads)
        else:
            optimizer(grads)
        return loss

    run_step = default_run_step_for_obj_net if for_object_net else default_run_step

    if jit:
        run_step = ms_jit(run_step)

    return run_step
