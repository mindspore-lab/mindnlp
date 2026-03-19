# Copyright (c) MindNLP Wizard contributors.
# Licensed under the Apache License, Version 2.0.

"""MindSpore model backend for lm-evaluation-harness.

Uses mindnlp.transformers (MindSpore) for model loading and inference on Ascend NPU.
Registered as model="mindspore" or model="mindnlp" in lm-eval.

IMPORTANT: mindnlp must be imported BEFORE lm_eval so mindtorch properly replaces torch.
After importing mindnlp, patch torch.utils.collect_env.get_pretty_env_info for lm_eval compat.
"""
import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

eval_logger = logging.getLogger(__name__)


def _pad_sequences(sequences: List[List[int]], pad_value: int = 0,
                   padding_side: str = "left") -> Tuple[np.ndarray, np.ndarray]:
    max_len = max(len(s) for s in sequences)
    padded = np.full((len(sequences), max_len), pad_value, dtype=np.int64)
    mask = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, s in enumerate(sequences):
        if padding_side == "left":
            offset = max_len - len(s)
            padded[i, offset:] = s
            mask[i, offset:] = 1
        else:
            padded[i, :len(s)] = s
            mask[i, :len(s)] = 1
    return padded, mask


@register_model("mindspore", "mindnlp")
class MindSporeLM(TemplateLM):
    """lm-eval model backend using MindSpore / MindNLP on Ascend NPU."""

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        pretrained: str,
        dtype: str = "float16",
        batch_size: Union[int, str] = 1,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        add_bos_token: bool = False,
        prefix_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        import mindspore
        from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=no-name-in-module
        from transformers import AutoConfig

        self._batch_size = int(batch_size) if str(batch_size).isdigit() else 1
        self._max_length = max_length
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = prefix_token_id

        # Device must already be set via DEVICE_TARGET env var before import mindnlp.
        # We only verify here, not set — setting here is too late for mindtorch init.
        current_device = mindspore.get_context('device_target')
        if current_device != 'Ascend':
            eval_logger.warning(
                f"[MindSporeLM] device_target is '{current_device}', not 'Ascend'. "
                f"NPU will NOT be used. Set os.environ['DEVICE_TARGET']='Ascend' "
                f"BEFORE importing mindnlp to enable NPU.")
        eval_logger.info(f"[MindSporeLM] Loading model: {pretrained}, dtype={dtype}, device={current_device}")

        self._config = AutoConfig.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code)

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        import torch as _torch
        torch_dtype_map = {
            "float16": _torch.float16, "float32": _torch.float32,
            "bfloat16": _torch.bfloat16, "auto": "auto",
        }
        _torch_dtype = torch_dtype_map.get(dtype, "auto")

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained, torch_dtype=_torch_dtype, trust_remote_code=trust_remote_code)

        if current_device == 'Ascend':
            eval_logger.info("[MindSporeLM] Moving model to NPU...")
            self._model = self._model.npu()

        if "gemma" in getattr(self._config, "model_type", ""):
            self.add_bos_token = True

        self.vocab_size = self.tokenizer.vocab_size
        eval_logger.info(
            f"[MindSporeLM] Model loaded. type={getattr(self._config, 'model_type', '?')}, "
            f"layers={getattr(self._config, 'num_hidden_layers', '?')}, "
            f"vocab={self.vocab_size}")

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "Ascend"

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(self, string: str, left_truncate_len=None,
                   add_special_tokens=None) -> List[int]:
        special_tokens_kwargs = {}
        if add_special_tokens is None:
            special_tokens_kwargs = {"add_special_tokens": False or self.add_bos_token}
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm

        res = []

        def _collate(req):
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)
        batch_size = override_bs if override_bs is not None else self._batch_size

        chunks = list(
            _chunks_list(re_ord.get_reordered(), batch_size))

        pbar = tqdm(total=len(requests), disable=disable_tqdm,
                    desc="Running loglikelihood requests")

        for chunk in chunks:
            inps_list = []
            cont_toks_list = []
            inplens = []

            for _, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                inps_list.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(len(inp))

            padded_inps, attn_mask = _pad_sequences(
                inps_list,
                pad_value=self.tokenizer.pad_token_id,
                padding_side="right",
            )

            input_ids = torch.tensor(padded_inps, dtype=torch.long).npu()
            attention_mask = torch.tensor(attn_mask, dtype=torch.long).npu()
            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            multi_logits = F.log_softmax(logits.float(), dim=-1)

            for idx, ((request_str, ctx_tokens, _), inplen, cont_toks) in enumerate(
                zip(chunk, inplens, cont_toks_list)
            ):
                contlen = len(cont_toks)
                logits_row = multi_logits[idx, inplen - contlen: inplen, :]

                greedy_tokens = logits_row.argmax(dim=-1)
                cont_toks_t = torch.tensor(cont_toks, dtype=torch.long).npu()
                cont_logprobs = logits_row[torch.arange(contlen).npu(), cont_toks_t]

                logprob_sum = float(cont_logprobs.sum().cpu())
                is_greedy = bool((greedy_tokens == cont_toks_t).all().cpu())

                answer = (logprob_sum, is_greedy)
                res.append(answer)

                if request_str is not None:
                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        from tqdm import tqdm

        loglikelihoods = []
        for (string,) in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Running rolling loglikelihood",
        ):
            token_list = self.tok_encode(string)
            rolling_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=token_list,
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            windows_as_requests = [
                (None, ctx, cont) for ctx, cont in rolling_windows
            ]

            nlls = self._loglikelihood_tokens(
                windows_as_requests, disable_tqdm=True,
                override_bs=self._batch_size,
            )

            total_nll = sum(nll for nll, _ in nlls)
            loglikelihoods.append(total_nll)
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), total_nll)

        return loglikelihoods

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        from tqdm import tqdm

        res = []
        reqs = [req.args for req in requests]

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ord = utils.Reorderer(reqs, _collate)

        eos_str = self.tok_decode(self.eot_token_id, skip_special_tokens=False)

        pbar = tqdm(total=len(requests), disable=disable_tqdm,
                    desc="Running generate_until requests")

        for chunk in _chunks_list(re_ord.get_reordered(), self._batch_size):
            contexts = []
            gen_kwargs_list = []
            for context, gen_kwargs in chunk:
                contexts.append(context)
                gen_kwargs_list.append(gen_kwargs)

            kwargs = copy.deepcopy(gen_kwargs_list[0]) if isinstance(gen_kwargs_list[0], dict) else {}
            until = kwargs.pop("until", None) or []
            if eos_str and eos_str not in until:
                until.append(eos_str)
            max_gen_toks = kwargs.pop("max_gen_toks", self.max_gen_toks)
            kwargs.pop("do_sample", None)

            max_ctx_len = self.max_length - max_gen_toks
            assert max_ctx_len > 0

            add_special = {"add_special_tokens": False or self.add_bos_token}
            encodings = [
                self.tokenizer.encode(ctx, **add_special)[-max_ctx_len:]
                for ctx in contexts
            ]

            padded_inps, attn_mask = _pad_sequences(
                encodings, pad_value=self.tokenizer.pad_token_id, padding_side="left")

            import torch
            input_ids = torch.tensor(padded_inps, dtype=torch.long).npu()
            attention_mask = torch.tensor(attn_mask, dtype=torch.long).npu()

            eos_ids = []
            if self.tokenizer.eos_token_id is not None:
                eos_ids.append(self.tokenizer.eos_token_id)
            for special_tok in ["<|im_end|>", "<|endoftext|>"]:
                tok_id = self.tokenizer.convert_tokens_to_ids(special_tok)
                if isinstance(tok_id, int) and tok_id != self.tokenizer.unk_token_id and tok_id not in eos_ids:
                    eos_ids.append(tok_id)

            generation_kwargs = {
                "max_new_tokens": max_gen_toks,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if eos_ids:
                generation_kwargs["eos_token_id"] = eos_ids

            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )

            if isinstance(output_ids, tuple):
                output_ids = output_ids[0]

            # MindTorch NPU tensors may expose `.numpy()` but require CPU init.
            # Prefer robust conversion order to avoid assertion on device tensors.
            if hasattr(output_ids, "asnumpy"):
                output_ids_np = output_ids.asnumpy()
            elif hasattr(output_ids, "cpu") and hasattr(output_ids, "numpy"):
                output_ids_np = output_ids.cpu().numpy()
            elif hasattr(output_ids, "numpy"):
                try:
                    output_ids_np = output_ids.numpy()
                except Exception:
                    if hasattr(output_ids, "cpu"):
                        output_ids_np = output_ids.cpu().numpy()
                    else:
                        raise
            else:
                output_ids_np = np.array(output_ids)

            for i, context in enumerate(contexts):
                ctx_len = len(encodings[i])
                pad_offset = padded_inps.shape[1] - ctx_len
                gen_start = pad_offset + ctx_len

                if output_ids_np.ndim == 1:
                    cont_toks = output_ids_np[gen_start:].tolist()
                else:
                    cont_toks = output_ids_np[i, gen_start:].tolist()

                s = self.tokenizer.decode(cont_toks, skip_special_tokens=True)

                for term in until:
                    if len(term) > 0:
                        s = s.split(term)[0]

                res.append(s)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs_list[i]), s)
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        except Exception:
            chat_history = [m for m in chat_history if m["role"] != "system"]
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )


def _chunks_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]
