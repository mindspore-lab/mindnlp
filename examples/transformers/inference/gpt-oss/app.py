import os
import gradio as gr
import mindhf
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
)

# Text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.95
)

def chat_fn(history, user_input):
    prompt = user_input
    response = generator(prompt)[0]["generated_text"]
    history = history + [(user_input, response)]
    return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# GPT-OSS-20B Chat")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Prompt", placeholder="Text...")
    state = gr.State([])

    def submit_fn(user_message, history):
        history, _ = chat_fn(history, user_message)
        return history, history

    msg.submit(submit_fn, [msg, state], [chatbot, state])

demo.launch(server_name="0.0.0.0", server_port=7860)
