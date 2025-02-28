import mindspore
from mindspore._c_expression import disable_multi_thread
disable_multi_thread()
from mindnlp.transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from mindnlp.configs import set_pyboost, use_pyboost
from mindnlp.core import nn, Tensor
from mindnlp.core import no_grad


from mindnlp.configs import use_pyboost, set_pyboost
print('use_pyboost:', use_pyboost())  # 这里默认是False
mindspore.set_context(
    mode=mindspore.PYNATIVE_MODE,
    pynative_synchronize=True,
    device_target="Ascend",
    # mode=mindspore.GRAPH_MODE,
    # jit_config={"jit_level":"O2"},
    ascend_config={"precision_mode": "allow_mix_precision"})
print(mindspore.get_context("mode"))
# specify the path to the model
model_path = "/home/HwHiAiUser/Janus-Pro-1B"
print('start load processor')
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
    model_path)
tokenizer = vl_chat_processor.tokenizer
print('loaded processor')
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, ms_dtype=mindspore.float16
)
print('loaded processor and ckpt ')
# question = 'describe this image'
question = 'what is the animal in the image'
image = "./inpain_model_cat.png"
conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{question}",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
)
print('process inputs')
# # run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
print('prepare inputs')
with no_grad():
    # # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(
        outputs[0].asnumpy().tolist(), skip_special_tokens=True)
    print(f"{prepare_inputs['sft_format'][0]}", answer)
