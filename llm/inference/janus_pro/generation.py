import os
import PIL.Image
import mindspore
from mindspore._c_expression import disable_multi_thread
disable_multi_thread()
import mindspore as ms
import numpy as np
from mindnlp.core import ops
from mindnlp.transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import mindspore.context as context

from mindnlp.configs import use_pyboost, set_pyboost
set_pyboost(False)
print('use_pyboost:', use_pyboost())
mindspore.set_context(
    mode=mindspore.PYNATIVE_MODE,
    # max_device_memory="15GB",
    pynative_synchronize=True,
    device_target="Ascend", 
    # mode=mindspore.GRAPH_MODE,  
    # jit_config={"jit_level":"O2"}, 
    ascend_config={"precision_mode":"allow_mix_precision"})
print(mindspore.get_context("mode"))
# specify the path to the model
model_path = "/home/HwHiAiUser/Janus-Pro-1B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, ms_dtype=mindspore.float16
)
print('loaded processor and ckpt ')


conversation = [
    {
        "role": "<|User|>",
        "content": "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
        # "content": "sun under blue sky",
    },
    {"role": "<|Assistant|>", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag
from mindnlp.core import no_grad

# @torch.inference_mode()
with no_grad():
    def generate(
        mmgpt: MultiModalityCausalLM,
        vl_chat_processor: VLChatProcessor,
        prompt: str,
        temperature: float = 1,
        parallel_size: int = 1, #16,
        cfg_weight: float = 5,
        # image_token_num_per_image: int = 8,#576,
        image_token_num_per_image: int = 576,#576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = ms.Tensor(input_ids, dtype=ms.int64)

        tokens = ops.zeros(parallel_size*2, len(input_ids), dtype=ms.int32)
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens) #(parallel_size*2, len(input_ids) )

        generated_tokens = ops.zeros(parallel_size, image_token_num_per_image, dtype=ms.int32)

        for i in range(image_token_num_per_image): 
            print(f"generating token {i}")
            outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state # (parallel_size*2, len(input_ids), 2048)
            
            logits = mmgpt.gen_head(hidden_states[:, -1, :]) #取最后一个input_id送入gen_head=>(parallel_size*2, vocab_size)
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = ops.softmax(logits / temperature, dim=-1)

            next_token = ops.multinomial(probs, num_samples=1) # (parallel_size, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(axis=-1)

            next_token = ops.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1) # (parallel_size*2)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token) # (parallel_size*2, 2048)
            # print("img_embeds.shape:", img_embeds.shape)
            # print("img_embeds.dtype:", img_embeds.dtype)
            inputs_embeds = img_embeds.unsqueeze(dim=1) #(parallel_size*2, 2048)
            print("generated one token")

        if image_token_num_per_image==576:
            dec = mmgpt.gen_vision_model.decode_code(generated_tokens.astype(ms.int32), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        else:
            pad_last_token = generated_tokens[:,-1].unsqueeze(dim=1).tile((1, 576-image_token_num_per_image))
            cat_generated_tokens=ops.cat([generated_tokens, pad_last_token], dim=1) 
            print("cat_generated_tokens.shape:",cat_generated_tokens.shape) #(1,576)
            dec = mmgpt.gen_vision_model.decode_code(cat_generated_tokens.astype(ms.int32), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.astype(ms.float32).asnumpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        os.makedirs('generated_samples', exist_ok=True)
        for i in range(parallel_size):
            save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
            PIL.Image.fromarray(visual_img[i]).save(save_path)
    generate(
        vl_gpt,
        vl_chat_processor,
        prompt,
    )