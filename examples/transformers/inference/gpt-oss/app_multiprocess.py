import mindnlp
import mindspore
from mindnlp import core
from mindnlp.core import distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

import gradio as gr
from threading import Thread

# mindspore.set_context(pynative_synchronize=True)
dist.init_process_group('hccl')

rank = dist.get_rank()

MODEL_NAME = "openai/gpt-oss-20b"

# Loading the tokenizer and model from Hugging Face's model hub.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
)

system_prompt = "You are a helpful and friendly chatbot"

def build_input_from_chat_history(chat_history, msg: str):
    messages = [{'role': 'system', 'content': system_prompt}]
    for user_msg, ai_msg in chat_history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': ai_msg})
    messages.append({'role': 'user', 'content': msg})
    return messages

# Function to generate model predictions.
def predict(message, history):
    dist.barrier()
    # Formatting the input for the model.
    messages = build_input_from_chat_history(history, message)
    input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True
        )
    input_len = core.tensor(input_ids.shape[1])
    dist.broadcast(input_len, 0)
    dist.barrier()
    streamer = TextIteratorStreamer(tokenizer, timeout=1200, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids.to('npu'),
        streamer=streamer,
        max_new_tokens=1024,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Starting the generation in a separate thread.
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
            break
        yield partial_message


def aux_predict():
    dist.barrier()
    # Formatting the input for the model.
    input_len = core.tensor(0)
    dist.broadcast(input_len, 0)
    dist.barrier()

    input_ids = core.zeros((1, input_len.item()), device='npu', dtype=core.int64)
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=1024,
    )
    out = model.generate(**generate_kwargs)
    return out


if rank == 0:
    # Setting up the Gradio chat interface.
    gr.ChatInterface(
        predict,
        title="GPT-OSS-20B-Chat",
        description="问几个问题",
        examples=['你是谁？', '介绍一下华为公司']
    ).launch()  # Launching the web interface.
else:
    while True:
        aux_predict()
