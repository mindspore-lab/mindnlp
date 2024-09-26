import gradio as gr
import numpy as np
import mindspore
from mindnlp.transformers import pipeline


MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 3600  # limit to 1 hour YouTube files


pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    ms_dtype=mindspore.float16
)


def transcribe(stream, new_chunk):
    generate_kwargs = {"language": "zh", "task": "transcribe"}

    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    print(y)

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    if stream.shape[0] < (3 * 48000):
        return stream, None

    text = pipe({"sampling_rate": sr, "raw": y}, generate_kwargs=generate_kwargs)["text"]

    if str(text).endswith((".", "。", '?', "？", '!', "！", ":", "：")):
        stream = None
    return stream, text  # type: ignore

demo = gr.Interface(
    transcribe,
    ["state", gr.Audio(sources=["microphone"], streaming=True)],
    ["state", "text"],
    live=True,
)

if __name__ == "__main__":
    demo.launch()
