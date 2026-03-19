# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

import logging

from mindspore.common import initializer  # pylint: disable=import-error
import transformers

LOG = logging.getLogger(__name__)


def monkeypatch_lmeval_shuffle():
    """Monkeypatch lm_eval to shuffle the dataset after downloading."""
    import lm_eval.api.task  # pylint: disable=import-error

    if hasattr(lm_eval.api.task.Task, "_monkey_patched"):
        return

    _old_task_dl = lm_eval.api.task.Task.download

    def _dl_shuffled(self: lm_eval.api.task.Task, *args, **kwargs):
        _old_task_dl(self, *args, **kwargs)
        self.dataset = self.dataset.shuffle()

    lm_eval.api.task.Task.download = _dl_shuffled

    _old_ct_dl = lm_eval.api.task.ConfigurableTask.download

    def _ct_dl_shuffled(self, *args, **kwargs):
        _old_ct_dl(self, *args, **kwargs)
        self.dataset = self.dataset.shuffle()

    lm_eval.api.task.ConfigurableTask.download = _ct_dl_shuffled

    lm_eval.api.task.Task._monkey_patched = True
    print("monkey has been patched")


def monkeypatch_tqdm(lm_eval: bool = True, mergekit: bool = True):
    """Patch lm_eval & wizard to use Ray's tqdm for progress bars."""

    from ray.experimental.tqdm_ray import tqdm as tqdm_ray  # pylint: disable=import-error

    def _tqdm_wrap(iterable=None, disable: bool = False, **kwargs):
        if disable:
            if iterable is not None:
                return iterable
            return lambda x: x
        res = tqdm_ray(iterable=iterable, **kwargs, flush_interval_s=1.0)
        res.refresh()
        return res

    def _patch_lm_eval():
        import lm_eval  # pylint: disable=import-error

        if hasattr(lm_eval, "_mk_tqdm_patched"):
            return

        import lm_eval.api.metrics  # pylint: disable=import-error
        import lm_eval.api.model  # pylint: disable=import-error
        import lm_eval.api.task  # pylint: disable=import-error
        import lm_eval.models.huggingface  # pylint: disable=import-error
        import lm_eval.models.vllm_causallms  # pylint: disable=import-error

        for module in (
            lm_eval.models.huggingface,
            lm_eval.models.vllm_causallms,
            lm_eval.api.model,
            lm_eval.api.task,
            lm_eval.api.metrics,
        ):
            setattr(module, "tqdm", _tqdm_wrap)

        lm_eval._mk_tqdm_patched = True

    if lm_eval:
        _patch_lm_eval()

    if mergekit:
        del mergekit

        from .. import graph as wizard_graph
        from .. import merge as wizard_merge
        from .. import tokenizer as wizard_tokenizer

        fake_module = type("fake_module", (), {"tqdm": staticmethod(_tqdm_wrap)})()

        wizard_graph.tqdm = fake_module
        wizard_merge.tqdm = fake_module
        wizard_tokenizer.tqdm = fake_module


def monkeypatch_lmeval_vllm():
    import lm_eval.models.vllm_causallms  # pylint: disable=import-error

    lm_eval.models.vllm_causallms.VLLM.AUTO_MODEL_CLASS = (
        transformers.AutoModelForCausalLM
    )


class NoInit:
    """Context manager that disables weight initialization for faster model
    instantiation.  Patches ``mindspore.common.initializer`` entry-points
    used by transformers model construction so that allocated parameters are
    left uninitialised (they will be overwritten by the merge anyway).
    """

    def __enter__(self):
        def noop(*args, **kwargs):
            pass

        self._originals = {
            "kaiming_uniform_": getattr(initializer, "HeUniform", None),
            "uniform_": getattr(initializer, "Uniform", None),
            "normal_": getattr(initializer, "Normal", None),
        }

        try:
            import torch.nn.init as _init

            (k, u, n) = (
                _init.kaiming_uniform_,
                _init.uniform_,
                _init.normal_,
            )
            _init.kaiming_uniform_ = noop
            _init.uniform_ = noop
            _init.normal_ = noop
            self._torch_funcs = (k, u, n)
        except ImportError:
            self._torch_funcs = None

        transformers.modeling_utils._init_weights = False

    def __exit__(self, *args):
        if self._torch_funcs is not None:
            try:
                import torch.nn.init as _init

                (k, u, n) = self._torch_funcs
                _init.kaiming_uniform_ = k
                _init.uniform_ = u
                _init.normal_ = n
            except ImportError:
                LOG.debug(
                    "Torch is unavailable while leaving NoInit context; "
                    "skipping torch init restoration"
                )

        transformers.modeling_utils._init_weights = True
