## Tinyllama Chatbot Implementation with Gradio

We offer an easy way to interact with Tinyllama. This guide explains how to set up a local Gradio demo for a chatbot using TinyLlama.

### Requirements
* Python>=3.9
* MindSpore>=2.4
* MindNLP>=0.4
* Gradio>=4.13.0

### Installation
`pip install -r requirements.txt`

### Usage

`python app.py`

* After running it, open the local URL displayed in your terminal in your web browser. (For server setup, use SSH local port forwarding with the command: `ssh -L [local port]:localhost:[remote port] [username]@[server address]`.)
* Interact with the chatbot by typing questions or commands.


**Note:** If you are runing Tinyllama on OrangePi, please use the follow instruction to free memory first:

```bash
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

```bash
export TE_PARALLEL_COMPILER=1
export MAX_COMPILE_CORE_NUMBER=1
export MS_BUILD_PROCESS_NUM=1
export MAX_RUNTIME_CORE_NUMBER=1
# if use O2
export MS_ENABLE_IO_REUSE=1
```