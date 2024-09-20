import os
from math import floor
from typing import Optional

import mindspore
import gradio as gr
from mindnlp.transformers import pipeline
from mindnlp.transformers.pipelines.audio_utils import ffmpeg_read


# configuration
MODEL_NAME = "BELLE-2/Belle-distilwhisper-large-v2-zh"
BATCH_SIZE = 16
CHUNK_LENGTH_S = 15
FILE_LIMIT_MB = 1000

ms_dtype = mindspore.float16

# define the pipeline
pipe = pipeline(
    model=MODEL_NAME,
    chunk_length_s=CHUNK_LENGTH_S,
    batch_size=BATCH_SIZE,
    ms_dtype=ms_dtype,
)

pipe.model.config.forced_decoder_ids = (
  pipe.tokenizer.get_decoder_prompt_ids(
    language="zh", 
    task="transcribe"
  )
)


def format_time(start: Optional[float], end: Optional[float]):

    def _format_time(seconds: Optional[float]):
        if seconds is None:
            return "complete    "
        minutes = floor(seconds / 60)
        hours = floor(seconds / 3600)
        seconds = seconds - hours * 3600 - minutes * 60
        m_seconds = floor(round(seconds - floor(seconds), 3) * 10 ** 3)
        seconds = floor(seconds)
        return f'{hours:02}:{minutes:02}:{seconds:02}.{m_seconds:03}'

    return f"[{_format_time(start)}-> {_format_time(end)}]:"


def get_prediction(inputs, prompt: Optional[str]):
    generate_kwargs = {"language": "zh", "task": "transcribe"}
    if prompt:
        generate_kwargs['prompt_ids'] = pipe.tokenizer.get_prompt_ids(prompt, return_tensors='ms')
    prediction = pipe(inputs, return_timestamps=True, generate_kwargs=generate_kwargs)
    text = "".join([c['text'] for c in prediction['chunks']])
    text_timestamped = "\n".join([
        f"{format_time(*c['timestamp'])} {c['text']}" for c in prediction['chunks']
    ])
    return text, text_timestamped


def transcribe(inputs: str, prompt):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    with open(inputs, "rb") as f:
        inputs = f.read()
    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}
    return get_prediction(inputs, prompt)


demo = gr.Blocks()
mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources="microphone", type="filepath"),
        gr.Textbox(lines=1, placeholder="Prompt"),
    ],
    outputs=["text", "text"],
    # layout="horizontal",
    # theme="huggingface",
    title=f"Transcribe Audio with {os.path.basename(MODEL_NAME)}",
    description=f"Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the Belle-Whisper checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files of arbitrary length.",
    allow_flagging="never",
)

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources="upload", type="filepath", label="Audio file"),
        gr.Textbox(lines=1, placeholder="Prompt"),
    ],
    outputs=["text", "text"],
    # layout="horizontal",
    # theme="huggingface",
    title=f"Transcribe Audio with {os.path.basename(MODEL_NAME)}",
    description=f"Transcribe long-form microphone or audio inputs with the click of a button! Demo uses Belle-Whisper checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files of arbitrary length.",
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe, file_transcribe], ["Microphone", "Audio file"])

demo.queue(api_open=False, default_concurrency_limit=40).launch(show_api=False, show_error=True)
