#!/usr/bin/env python3
"""
Wrapper to run lm-eval-harness with Wizard MindSpore backend.

Registers the 'mindspore' model type via mindnlp.wizard.merge.eval.mindspore_lm,
then delegates to lm-eval's CLI.

Usage:
  python run_lm_eval.py --model mindspore \
      --model_args pretrained=./output/merged,dtype=bfloat16 \
      --tasks arc_challenge --num_fewshot 25 --batch_size 1
"""

import os
import sys


def main() -> int:
    os.environ.setdefault("DEVICE_TARGET", "Ascend")

    # mindnlp must be imported before torch so mindtorch proxies work.
    import mindnlp  # pylint: disable=unused-import
    import torch
    from torch.utils import collect_env

    if not hasattr(collect_env, "get_pretty_env_info"):
        collect_env.get_pretty_env_info = (
            lambda: "mindtorch environment info unavailable"
        )

    # Restore DynamicLayer.update for transformers 4.55+ compatibility.
    try:
        import transformers.cache_utils as _cache_mod

        _DL = getattr(_cache_mod, "DynamicLayer", None)
        if _DL is not None:

            def _dynamic_layer_update(self, key_states, value_states, cache_kwargs=None):
                if self.keys is None:
                    self.keys = key_states
                    self.values = value_states
                else:
                    self.keys = torch.cat([self.keys, key_states], dim=-2)
                    self.values = torch.cat([self.values, value_states], dim=-2)
                return self.keys, self.values

            _DL.update = _dynamic_layer_update
    except Exception as _e:
        print(f"[warn] DynamicLayer.update restore skipped: {_e}")

    # Force eager attention to avoid mindtorch SDPA reshape bug.
    try:
        from mindnlp.transformers import AutoModelForCausalLM as _AMCLM  # pylint: disable=no-name-in-module

        _orig_from_pretrained = _AMCLM.from_pretrained

        @classmethod
        def _from_pretrained_eager(cls, *args, **kwargs):
            kwargs.setdefault("attn_implementation", "eager")
            return _orig_from_pretrained.__func__(cls, *args, **kwargs)

        _AMCLM.from_pretrained = _from_pretrained_eager
    except Exception as _e:
        print(f"[warn] eager-attn patch skipped: {_e}")

    # Register lm-eval model type: "mindspore" / "mindnlp".
    import mindnlp.wizard.merge.eval.mindspore_lm  # pylint: disable=unused-import
    from lm_eval.__main__ import cli_evaluate

    sys.argv = ["lm-eval", *sys.argv[1:]]
    rc = cli_evaluate()  # pylint: disable=assignment-from-no-return
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
