from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("sunzeyeah/pangu-350M-sft")
model = AutoModelForCausalLM.from_pretrained("sunzeyeah/pangu-350M-sft")

prompt = "我不能确定对方是不是喜欢我,我却想分分秒秒跟他在一起,有谁能告诉我如何能想他少一点<sep>回答："
inputs = tokenizer(prompt, add_special_tokens=False, return_token_type_ids=False, return_tensors="ms")
outputs = model.generate(**inputs,
                         max_new_tokens=100,
                         pad_token_id=tokenizer.pad_token_id,
                         do_sample=True,
                         num_return_sequences=1,
                         top_p=0.8,
                         temperature=0.8)
results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
results = [result.split("答:", maxsplit=1)[1] for result in results]
print(results)
