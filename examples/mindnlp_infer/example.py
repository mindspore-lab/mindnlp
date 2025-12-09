from mindnlp.inference import LLM, SamplingParams

if __name__ == '__main__':
    # MODEL_PATH = "/YOUR/MODEL/PATH"
    MODEL_PATH = "Qwen/Qwen3-0.6B"
    llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = ["Hello, Nano-vLLM."]
    outputs = llm.generate(prompts, sampling_params)
    print('decoded outputs:', outputs[0]["text"])