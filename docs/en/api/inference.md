# Inference

This page documents the lightweight inference stack under `mindnlp.inference`, aimed at running local LLM checkpoints with tiled KV-cache management and tensor-parallel workers.

## Public entrypoints

- `LLM` (`mindnlp.inference.LLM`): a thin alias of `LLMEngine`, orchestrating worker processes, tokenizer, scheduler and generation loop.
- `Config` (`mindnlp.inference.config.Config`): dataclass describing runtime limits and hardware settings. Key fields include:
  - `model` (path to a local Hugging Face-style checkpoint; required)
  - batching: `max_num_batched_tokens` (16384), `max_num_seqs` (512), `max_model_len` (capped by HF config)
  - parallelism: `tensor_parallel_size` (1..8), `gpu_memory_utilization` (0.9), `enforce_eager` (fallback when CUDA graphs are undesired)
  - KV cache: `kvcache_block_size` (multiple of 256), `num_kvcache_blocks` (auto derived), `eos` (filled from tokenizer)
- `SamplingParams` (`mindnlp.inference.sampling_params.SamplingParams`): sampling controls, currently `temperature`, `max_tokens`, and `ignore_eos`. Temperatures below `1e-10` are rejected to avoid greedy sampling.

## Engine architecture

- **LLMEngine** (`mindnlp.inference.engine.llm_engine.LLMEngine`):
  - Spawns one or more `ModelRunner` processes (role `MS_WORKER`/`MS_SCHED`) with MindTorch distributed initialized over TCP.
  - Builds a `Scheduler` to batch prefill/decode steps, and an AutoTokenizer for prompt encoding/decoding.
  - User-facing methods:
    - `add_request(prompt, sampling_params)` queues a prompt (string or token list).
    - `step()` performs a scheduler tick, runs model forward, and returns finished sequence outputs plus token throughput counters.
    - `generate(prompts, sampling_params, use_tqdm=True)` high-level helper returning decoded strings and token ids.
    - `is_finished()` and `exit()` lifecycle helpers.
- **Scheduler & Sequence** (`mindnlp.inference.engine.scheduler`, `.sequence`):
  - `Scheduler` maintains `waiting` and `running` queues, deciding between prefill and decode phases per step.
  - `Sequence` tracks token ids, cached blocks, sampling params, and completion status; exposes prompt vs completion slices.
  - `BlockManager` uses content hashes (xxhash) to allocate/deallocate KV-cache blocks, enabling prefix reuse and preemption.
- **ModelRunner** (`mindnlp.inference.engine.model_runner.ModelRunner`):
  - Loads `Qwen3ForCausalLM` via `load_model`, configures KV-cache tensors, and optionally captures CUDA graphs for decode throughput.
  - Prepares inputs for prefill/decode (including block tables, slot mappings) and runs the sampler to select next tokens on rank 0.
  - Communicates across ranks through shared memory events when tensor parallelism > 1, and cleans up on `exit()`.
- **Layers & utils**: helper modules under `mindnlp.inference.layers` (e.g., `sampler`, `attention`, `activation`, `rotary_embedding`, etc.) and `mindnlp.inference.utils` (context management, model loading).

## Basic usage

```python
from mindnlp.inference import LLM, SamplingParams

# Point to a local Qwen3-style HF checkpoint directory
llm = LLM(
    model="/path/to/qwen3-checkpoint",
    tensor_parallel_size=1,
    max_num_batched_tokens=8192,
)

sampling = SamplingParams(temperature=0.7, max_tokens=128)
prompts = ["Hello, who are you?", "Write a short poem about MindNLP."]

outputs = llm.generate(prompts, sampling, use_tqdm=True)
for out in outputs:
    print(out["text"])

llm.exit()  # optional explicit shutdown
```

## Operational notes

- Only local checkpoints are supported today; `Config.__post_init__` asserts `model` is an existing directory and reads `AutoConfig`.
- Default `max_model_len` is bounded by the checkpoint’s `max_position_embeddings`.
- GPU memory usage is inferred at runtime to size KV-cache blocks; ensure adequate free memory before starting workers.
- Current `ModelRunner` is wired to `Qwen3ForCausalLM`; extending to other architectures requires updating the loader/imports.

## Roadmap ideas

- Add auto model/architecture selection instead of the current Qwen3-only path.
- Expose streaming/token-level callbacks from `LLMEngine.step` to integrate with services.
- Expand `SamplingParams` (top-k/p, repetition penalties) and wire them into `Sampler`.
- Document and ship ready-made configs for common hardware (single GPU vs multi-GPU tensor parallel).
