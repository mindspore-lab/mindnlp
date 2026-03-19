import json
import os
import tempfile
from typing import Callable, Dict, List, Optional, Union

import pytest
import tokenizers
import numpy as np
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

from mindnlp.wizard.merge.config import InputModelDefinition, MergeConfiguration
from mindnlp.wizard.merge.io.lazy_tensor_loader import LazyTensorLoader
from mindnlp.wizard.merge.merge import MergeOptions, run_merge
from mindnlp.wizard.merge.tokenizer import TokenizerConfig


@pytest.fixture(scope="session")
def model_base(tmp_path_factory):
    model_path = tmp_path_factory.mktemp("wizard_model_base")
    _make_picollama(str(model_path), vocab_size=64)
    _make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(str(model_path))
    return str(model_path)


@pytest.fixture(scope="session")
def model_chatml(tmp_path_factory):
    model_path = tmp_path_factory.mktemp("wizard_model_chatml")
    _make_picollama(str(model_path), vocab_size=66)
    tok = _make_tokenizer(
        vocab_size=64,
        added_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tok.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}"
        "{% endfor %}"
    )
    tok.save_pretrained(str(model_path))
    return str(model_path)


@pytest.fixture(scope="session")
def model_padded(tmp_path_factory):
    model_path = tmp_path_factory.mktemp("wizard_model_padded")
    _make_picollama(str(model_path), vocab_size=64)
    _make_tokenizer(
        vocab_size=64,
        added_tokens=["<UNUSED_0>", "<UNUSED_1>", "<UNUSED_2>", "<UNUSED_3>"],
    ).save_pretrained(str(model_path))
    return str(model_path)


def _make_picollama(path: str, vocab_size: int = 64):
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=4,
        num_hidden_layers=2,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(path, safe_serialization=True)
    return path


def _make_tokenizer(
    vocab_size: int,
    added_tokens: List[Union[str, tokenizers.AddedToken]],
) -> LlamaTokenizerFast:
    tokens = ["<unk>", "<s>", "</s>"] + [f"_tok_{idx}" for idx in range(3, vocab_size)]
    tokens = tokens[:vocab_size]
    tok_data = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": dict(zip(tokens, range(vocab_size))),
            "merges": [],
        },
        "added_tokens": [],
    }
    tok = tokenizers.Tokenizer.from_str(json.dumps(tok_data))
    with tempfile.TemporaryDirectory() as p:
        tok_path = os.path.join(p, "tokenizer.json")
        tok.save(tok_path)
        res = LlamaTokenizerFast(tokenizer_file=tok_path)
    res.add_tokens(added_tokens)
    return res


def _check_tokenizer(
    model_path: str,
    expected_size: int,
    expected_added_ct: Optional[int] = None,
    must_contain: Optional[List[str]] = None,
    must_not_contain: Optional[List[str]] = None,
):
    tok = LlamaTokenizerFast.from_pretrained(model_path)
    vocab = tok.get_vocab()
    assert len(vocab) == expected_size
    if expected_added_ct is not None:
        assert len(tok.added_tokens_decoder) == expected_added_ct
    if must_contain:
        for token in must_contain:
            assert token in vocab
    if must_not_contain:
        for token in must_not_contain:
            assert token not in vocab


def _run_and_check_merge(
    config: MergeConfiguration,
    validate: Optional[Callable[[str], None]] = None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        run_merge(config, out_path=tmpdir, options=MergeOptions())
        assert os.path.exists(os.path.join(tmpdir, "config.json"))
        assert (
            os.path.exists(os.path.join(tmpdir, "model.safetensors.index.json"))
            or os.path.exists(os.path.join(tmpdir, "model.safetensors"))
        )
        if validate:
            validate(tmpdir)


class _ModelEmbeddings:
    def __init__(self, model_path: str):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
        self.vocab = self.tokenizer.get_vocab()
        loader = LazyTensorLoader.from_disk(model_path, lazy_loader=False)
        self.embed_tokens = loader.get_tensor("model.embed_tokens.weight")

    def token_embedding(self, token: str):
        idx = self.vocab.get(token)
        if idx is None:
            return None
        return self.embed_tokens[idx, :]


class TestTokenizerMergekitParity:
    def _make_config(
        self,
        model_paths: List[str],
        *,
        base_model: Optional[str] = None,
        tokenizer_source=None,
        chat_template=None,
        merge_method: str = "linear",
        t: Optional[float] = None,
        tokenizer_config: Optional[TokenizerConfig] = None,
    ) -> MergeConfiguration:
        cfg = {
            "merge_method": merge_method,
            "models": [
                {"model": mp, "parameters": {"weight": 0.5}}
                for mp in model_paths
            ],
            "dtype": "float16",
        }
        if base_model:
            cfg["base_model"] = base_model
        if tokenizer_source is not None:
            cfg["tokenizer_source"] = tokenizer_source
        if chat_template is not None:
            cfg["chat_template"] = chat_template
        if tokenizer_config is not None:
            cfg["tokenizer"] = tokenizer_config.model_dump(mode="json")
        if t is not None:
            cfg["parameters"] = {"t": t}
        return MergeConfiguration.model_validate(cfg)

    def test_tokenizer_source_model_matches_mergekit_behavior(
        self, model_base: str, model_chatml: str
    ):
        config = self._make_config(
            [model_base, model_chatml],
            base_model=model_base,
            tokenizer_config=TokenizerConfig(source=model_chatml),
        )

        def _validate(model_path: str):
            _check_tokenizer(
                model_path,
                expected_size=66,
                must_contain=["<|im_start|>", "<|im_end|>"],
            )

        _run_and_check_merge(config, validate=_validate)

    def test_legacy_mode_uses_base_tokenizer(
        self, model_base: str, model_padded: str, model_chatml: str
    ):
        config = self._make_config(
            [model_base, model_padded, model_chatml],
            base_model=model_base,
        )

        def _validate(model_path: str):
            _check_tokenizer(
                model_path,
                expected_size=64,
                expected_added_ct=3,
            )

        _run_and_check_merge(config, validate=_validate)

    def test_tokenizer_source_base_matches_legacy(
        self, model_base: str, model_padded: str, model_chatml: str
    ):
        config = self._make_config(
            [model_base, model_padded, model_chatml],
            base_model=model_base,
            tokenizer_source="base",
        )

        def _validate(model_path: str):
            _check_tokenizer(
                model_path,
                expected_size=64,
                expected_added_ct=3,
            )

        _run_and_check_merge(config, validate=_validate)

    def test_tokenizer_source_union_drops_unused_and_keeps_chat_tokens(
        self, model_base: str, model_padded: str, model_chatml: str
    ):
        config = self._make_config(
            [model_base, model_padded, model_chatml],
            base_model=model_base,
            tokenizer_source="union",
        )

        def _validate(model_path: str):
            _check_tokenizer(
                model_path,
                expected_size=66,
                expected_added_ct=5,
                must_contain=["<|im_start|>", "<|im_end|>"],
                must_not_contain=[f"<UNUSED_{idx}>" for idx in range(4)],
            )
            emb_out = _ModelEmbeddings(model_path)
            emb_chatml = _ModelEmbeddings(model_chatml)
            np.testing.assert_allclose(
                emb_out.token_embedding("<|im_start|>").asnumpy(),
                emb_chatml.token_embedding("<|im_start|>").asnumpy(),
                atol=2e-5,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                emb_out.token_embedding("<|im_end|>").asnumpy(),
                emb_chatml.token_embedding("<|im_end|>").asnumpy(),
                atol=1e-3,
                rtol=1e-4,
            )

        _run_and_check_merge(config, validate=_validate)

    def test_chat_template_auto_is_saved(
        self, model_base: str, model_chatml: str
    ):
        config = self._make_config(
            [model_base, model_chatml],
            base_model=model_base,
            tokenizer_config=TokenizerConfig(source=model_chatml),
            chat_template="auto",
        )

        def _validate(model_path: str):
            tok = LlamaTokenizerFast.from_pretrained(model_path)
            assert tok.chat_template is not None
            assert "<|im_start|>" in tok.chat_template

        _run_and_check_merge(config, validate=_validate)

    def test_slerp_with_union_tokenizer_works(
        self, model_base: str, model_chatml: str
    ):
        config = self._make_config(
            [model_base, model_chatml],
            base_model=model_base,
            tokenizer_source="union",
            merge_method="slerp",
            t=0.5,
        )

        def _validate(model_path: str):
            _check_tokenizer(
                model_path,
                expected_size=66,
                must_contain=["<|im_start|>", "<|im_end|>"],
            )

        _run_and_check_merge(config, validate=_validate)

    def test_force_token_sources(
        self, model_base: str, model_chatml: str
    ):
        config = self._make_config(
            [model_base, model_chatml],
            base_model=model_base,
            merge_method="linear",
            tokenizer_config=TokenizerConfig(
                source="union",
                tokens={
                    "_tok_10": {"source": model_chatml, "force": True},
                    "_tok_11": {"source": model_base, "force": True},
                },
            ),
        )

        def _validate(model_path: str):
            _check_tokenizer(
                model_path,
                expected_size=66,
                must_contain=["<|im_start|>", "<|im_end|>"],
            )
            emb_out = _ModelEmbeddings(model_path)
            emb_base = _ModelEmbeddings(model_base)
            emb_chatml = _ModelEmbeddings(model_chatml)

            np.testing.assert_allclose(
                emb_out.token_embedding("_tok_10").asnumpy(),
                emb_chatml.token_embedding("_tok_10").asnumpy(),
                atol=2e-5,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                emb_out.token_embedding("_tok_11").asnumpy(),
                emb_base.token_embedding("_tok_11").asnumpy(),
                atol=2e-5,
                rtol=1e-4,
            )

        _run_and_check_merge(config, validate=_validate)

    def test_model_token_source_variants(
        self, model_base: str, model_chatml: str
    ):
        config = self._make_config(
            [model_base, model_chatml],
            base_model=model_base,
            merge_method="linear",
            tokenizer_config=TokenizerConfig(
                source="base",
                tokens={
                    "_tok_20": {
                        "source": {
                            "kind": "model_token",
                            "model": model_chatml,
                            "token_id": 64,
                        },
                        "force": True,
                    },
                    "_tok_21": {
                        "source": {
                            "kind": "model_token",
                            "model": model_base,
                            "token": "<s>",
                        },
                        "force": True,
                    },
                },
            ),
        )

        def _validate(model_path: str):
            _check_tokenizer(model_path, expected_size=64)
            emb_out = _ModelEmbeddings(model_path)
            emb_base = _ModelEmbeddings(model_base)
            emb_chatml = _ModelEmbeddings(model_chatml)

            np.testing.assert_allclose(
                emb_out.token_embedding("_tok_20").asnumpy(),
                emb_chatml.embed_tokens[64, :].asnumpy(),
                atol=2e-5,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                emb_out.token_embedding("_tok_21").asnumpy(),
                emb_base.token_embedding("<s>").asnumpy(),
                atol=2e-5,
                rtol=1e-4,
            )

        _run_and_check_merge(config, validate=_validate)

    def test_pad_to_multiple_of_updates_vocab_and_embedding(
        self, model_chatml: str
    ):
        config = MergeConfiguration.model_validate(
            {
                "merge_method": "linear",
                "base_model": model_chatml,
                "models": [
                    {"model": model_chatml, "parameters": {"weight": 1.0}},
                    {"model": model_chatml, "parameters": {"weight": 0.0}},
                ],
                "dtype": "float16",
                "tokenizer": {
                    "source": "base",
                    "pad_to_multiple_of": 16,
                },
            }
        )
        real_vocab_size = 64 + 2
        padded_size = (real_vocab_size // 16 + 1) * 16

        def _validate(model_path: str):
            cfg = LlamaConfig.from_pretrained(model_path)
            assert cfg.vocab_size == padded_size
            _check_tokenizer(
                model_path,
                expected_size=real_vocab_size,
                must_contain=["<|im_start|>", "<|im_end|>"],
            )
            emb_out = _ModelEmbeddings(model_path)
            assert emb_out.embed_tokens.shape[0] == padded_size

        _run_and_check_merge(config, validate=_validate)

    def test_chat_template_builtin_name_is_saved(
        self, model_base: str, model_chatml: str
    ):
        config = self._make_config(
            [model_base, model_chatml],
            base_model=model_base,
            merge_method="linear",
            chat_template="chatml",
        )

        def _validate(model_path: str):
            tok = LlamaTokenizerFast.from_pretrained(model_path)
            assert tok.chat_template
            assert "<|im_start|>" in tok.chat_template

        _run_and_check_merge(config, validate=_validate)

    def test_chat_template_literal_jinja_is_saved(
        self, model_base: str, model_chatml: str
    ):
        literal_template = "{{messages[0]['content']}}"
        config = self._make_config(
            [model_base, model_chatml],
            base_model=model_base,
            merge_method="linear",
            chat_template=literal_template,
        )

        def _validate(model_path: str):
            tok = LlamaTokenizerFast.from_pretrained(model_path)
            assert tok.chat_template
            assert literal_template in tok.chat_template

        _run_and_check_merge(config, validate=_validate)
