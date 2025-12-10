import gradio as gr
import time
import numpy as np
import mindspore
from mindhf.transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

ms_dtype = mindspore.float16
MODEL_NAME = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME, ms_dtype=ms_dtype, low_cpu_mem_usage=True
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    ms_dtype=ms_dtype,
)

prompt = "以下是普通话的句子。" # must have periods
prompt_ids = processor.get_prompt_ids(prompt, return_tensors="ms")
generate_kwargs = {"prompt_ids": prompt_ids}

def transcribe(inputs, previous_transcription):
    start_time = time.time() 
    try:
        sample_rate, audio_data = inputs
        audio_data = audio_data.astype(np.float32)
        audio_data /= np.max(np.abs(audio_data))

        transcription = pipe({"sampling_rate": sample_rate, "raw": audio_data}, generate_kwargs=generate_kwargs)["text"]
        previous_transcription += transcription

        end_time = time.time()
        latency = end_time - start_time
        return previous_transcription, f"{latency:.2f}"
    except Exception as e:
        print(f"Error during Transcription: {e}")
        return previous_transcription, "Error"


def clear():
    return ""

with gr.Blocks() as microphone:
    with gr.Column():
        gr.Markdown(f"# Realtime Whisper Large V3 Turbo: \n Transcribe Audio in Realtime. This Demo uses the Checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers.\n Note: The first token takes about 5 seconds. After that, it works flawlessly.")
        with gr.Row():
            input_audio_microphone = gr.Audio(streaming=True)
            output = gr.Textbox(label="Transcription", value="")
            latency_textbox = gr.Textbox(label="Latency (seconds)", value="0.0", scale=0)
        with gr.Row():
            clear_button = gr.Button("Clear Output")

        input_audio_microphone.stream(transcribe, [input_audio_microphone, output], [output, latency_textbox], time_limit=45, stream_every=2, concurrency_limit=None)
        clear_button.click(clear, outputs=[output])

with gr.Blocks() as file:
    with gr.Column():
        gr.Markdown(f"# Realtime Whisper Large V3 Turbo: \n Transcribe Audio in Realtime. This Demo uses the Checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers.\n Note: The first token takes about 5 seconds. After that, it works flawlessly.")
        with gr.Row():
            input_audio_microphone = gr.Audio(sources="upload", type="numpy")
            output = gr.Textbox(label="Transcription", value="")
            latency_textbox = gr.Textbox(label="Latency (seconds)", value="0.0", scale=0)
        with gr.Row():
            submit_button = gr.Button("Submit")
            clear_button = gr.Button("Clear Output")

        submit_button.click(transcribe, [input_audio_microphone, output], [output, latency_textbox], concurrency_limit=None)
        clear_button.click(clear, outputs=[output])

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.TabbedInterface([microphone, file], ["Microphone", "Transcribe from file"])

demo.launch()