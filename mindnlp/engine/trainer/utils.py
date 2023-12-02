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

from mindspore import ops, value_and_grad

from mindnlp import ms_jit
from mindnlp._legacy.amp import all_finite, init_status
from mindnlp.utils import ModelOutput

def get_default_forward_fn_with_loss_fn(network, loss_fn, loss_scaler):
    """get default forward function with loss function"""
    # forward function
    def forward_fn(labels, *args, **kwargs):
        logits_list = ()
        logits = network(*args, **kwargs)
        if isinstance(logits, tuple):
            logits_list += logits
        elif isinstance(logits, ModelOutput):
            logits_list += (logits.logits,)
        else:
            logits_list += (logits,)

        logits_list += labels
        loss = loss_fn(*logits_list)
        loss = loss_scaler.scale(loss)
        return loss

    return forward_fn

def get_default_forward_fn_without_loss_fn(network, loss_scaler):
    """get default forward function without loss function"""
    def forward_fn(*args, **kwargs):
        outputs_list = ()
        outputs = network(*args, **kwargs)
        if isinstance(outputs, tuple):
            outputs_list += outputs
        elif isinstance(outputs, ModelOutput):
            outputs_list += (outputs.loss,)
        else:
            outputs_list += (outputs,)

        loss = loss_scaler.scale(outputs_list[0])
        return loss

    return forward_fn

def get_default_train_step_fn(forward_fn, optimizer, loss_scaler, check_gradients, jit, for_object_net=False):
    """get default train function"""
    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    def default_run_step(labels, *args, **kwargs):
        """Core process of each step, including the forward propagation process and back propagation of data."""
        status = init_status()
        loss, grads = grad_fn(labels, *args, **kwargs)
        loss = loss_scaler.unscale(loss)
        if check_gradients:
            is_finite = all_finite(grads, status)
            if is_finite:
                grads = loss_scaler.unscale(grads)
                optimizer(grads)
            loss_scaler.adjust(is_finite)
        else:
            optimizer(grads)
        return loss

    def default_run_step_for_obj_net(*args, **kwargs):
        """Core process of each step, including the forward propagation process and back propagation of data."""
        status = init_status()
        args = ops.depend(args, status)
        loss, grads = grad_fn(*args, **kwargs)
        loss = loss_scaler.unscale(loss)
        if check_gradients:
            is_finite = all_finite(grads, status)
            if is_finite:
                grads = loss_scaler.unscale(grads)
                loss = ops.depend(loss, optimizer(grads))
            loss = ops.depend(loss, loss_scaler.adjust(is_finite))
        else:
            loss = ops.depend(loss, optimizer(grads))
        return loss

    run_step = default_run_step_for_obj_net if for_object_net else default_run_step

    if jit:
        run_step = ms_jit(run_step)

    return run_step
