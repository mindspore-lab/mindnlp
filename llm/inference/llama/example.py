# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import time
import json
import fire
from pathlib import Path

import mindspore
from mindspore.communication import init, get_rank, get_group_size

from llama.model import LlamaConfig, Transformer
from llama.tokenizer import Tokenizer
from llama.generation import LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    rank_id = get_rank()
    rank_size = get_group_size()

    # seed must be the same in all processes
    mindspore.set_seed(1234)
    return rank_id, rank_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    rank_id: int,
    rank_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.ckpt"))
    assert rank_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {rank_size}"
    ckpt_path = str(checkpoints[rank_id])
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: LlamaConfig = LlamaConfig(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    print("Instanctial model")
    model = Transformer(model_args)
    print("Loading checkpoint")
    mindspore.load_checkpoint(ckpt_path, net=model)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    rank_id, rank_size = setup_model_parallel()
    if rank_id > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, rank_id, rank_size, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://hf-mirror.com/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    init(backend_name="nccl")
    fire.Fire(main)
