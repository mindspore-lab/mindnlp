import argparse
import mindspore
from mindspore.nn import AdamWeightDecay
from squad_dataset import get_squad_dataset
from mindnlp.peft import LoraConfig, get_peft_model
from mindnlp.transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)

mindspore.set_context(device_target="CPU")


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path)

    ds = get_squad_dataset(tokenizer, args.batch_size)
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias='none',
        task_type="QUESTION_ANSWER",
        target_modules=args.lora_target_modules.split(","),
    )
    model = get_peft_model(model=model, peft_config=peft_config)
    # model.print_trainable_parameters()

    optimizer = AdamWeightDecay(
        params=model.trainable_params(), learning_rate=args.lr)

    def forward_fn(input_ids, token_type_ids, attention_mask, start_positions, end_positions):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )
        return output.loss

    grad_fn = mindspore.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False
    )

    total_loss, total_step = 0, 0
    for _, (input_ids, token_type_ids, attention_mask, start_positions, end_positions) in enumerate(ds):
        (loss), grad = grad_fn(input_ids, token_type_ids,
                               attention_mask, start_positions, end_positions)
        optimizer(grad)
        total_loss += loss.asnumpy()
        total_step += 1
        curr_loss = total_loss / total_step
        print({"train-loss": f"{curr_loss:.2f}"})

    model.save_pretrained(save_directory=args.model_save_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--model_name_or_path", default="YituTech/conv-bert-base",
                        type=str, help="YituTech/conv-bert-base")
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float,
                        help="Set 2e-5 for full-finetuning.")
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--lora_target_modules", type=str,
                        default="query, key, value,conv_out_layer, conv_kernel_layer, dense")
    parser.add_argument("--model_save_dir", type=str,
                        default="convbert_lora_peft")
    args = parser.parse_args()
    main(args)
