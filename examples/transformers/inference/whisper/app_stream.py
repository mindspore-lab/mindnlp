import queue
import threading
import time

import gradio as gr
import numpy as np
import mindspore
from mindspore.dataset.audio import Resample
from mindhf.transformers import pipeline, AutoProcessor
from silero_vad_mindspore import load

MODEL_NAME = "openai/whisper-large-v3"
THRESH_HOLD = 0.5

stream_queue = queue.Queue()

vad_model = load('silero_vad_v4')

processor = AutoProcessor.from_pretrained(MODEL_NAME)
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    ms_dtype=mindspore.float16
)

prompt = "以下是普通话的句子。" # must have periods
prompt_ids = processor.get_prompt_ids(prompt, return_tensors="ms")

text = ""
silence_count = 0

resample = Resample(48000, 16000)
generate_kwargs = {"language": "zh", "task": "transcribe", "prompt_ids": prompt_ids}
                #    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), "no_speech_threshold": 0.5, "logprob_threshold": -1.0}

# warm up
random_sample = np.random.randn(16000).astype(np.float32)
vad_model(mindspore.tensor(random_sample), 16000)
pipe(random_sample, generate_kwargs=generate_kwargs, return_timestamps='word')

def pipeline_consumer():
    global text
    while True:
        chunk = stream_queue.get()
        # print(speech_score)
        genreated_text = pipe(chunk, generate_kwargs=generate_kwargs, return_timestamps='word')["text"]
        text += genreated_text + '\n'

        stream_queue.task_done()
 
        if stream_queue.empty() and stream_queue.unfinished_tasks == 0:
            time.sleep(1)


def transcribe(stream, new_chunk):
    global text

    sr, y = new_chunk

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    # print('sample shape:', y.shape)
    speech_score = vad_model(mindspore.tensor(y), sr)
    speech_score = speech_score.item()
    print('speech socre', speech_score)

    if speech_score > 0.5:
        if stream is not None:
            if stream.shape < y.shape or (stream[-len(y):] - y).sum() != 0:
                stream = np.concatenate([stream, y])
        else:
            stream = y

    if stream is not None and stream.shape[0] >= (48000 * 5): # 5s if continue talk
        print('stream shape:', stream.shape)
        stream_queue.put({"sampling_rate": sr, "raw": stream})
        stream = None

    return stream, text  # type: ignore

input_audio = gr.Audio(sources=["microphone"], streaming=True)
demo = gr.Interface(
    transcribe,
    ["state", input_audio],
    ["state", "text"],
    live=True,
)

if __name__ == "__main__":
    c = threading.Thread(target=pipeline_consumer)
    c.start()
    demo.launch()
