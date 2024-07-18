import argparse
import mindspore
from mindspore import nn
from mindspore.nn import AdamWeightDecay
from squad_dataset import get_squad_dataset
from mindnlp.transformers import (
    AutoTokenizer,
    MSConvBertForQuestionAnswering,
)
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import Model, LossMonitor

MODEL_PATH = "/data2/neoming_convbert/dev/mindnlp/llm/parallel/convbert/.mindnlp/model/YituTech/conv-bert-base"


def parallel_setup():
    init("nccl")
    device_num = get_group_size()
    rank = get_rank()
    print("rank_id is {}, device_num is {}".format(rank, device_num))
    mindspore.reset_auto_parallel_context()
    mindspore.set_auto_parallel_context(
        parallel_mode=mindspore.ParallelMode.DATA_PARALLEL,
    )
    mindspore.set_context(mode=mindspore.GRAPH_MODE)
    mindspore.set_context(device_target="GPU")


class ConvBertTrainNet(nn.Cell):
    def __init__(self, backbone):
        super(ConvBertTrainNet, self).__init__(auto_prefix=False)
        self._backbone = backbone

    def construct(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        start_positions,
        end_positions,
    ):
        loss = self._backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        return loss

    @property
    def backbone_network(self):
        return self._backbone


def main(args):
    parallel_setup()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = MSConvBertForQuestionAnswering.from_pretrained(
        args.model_name_or_path)
    ds = get_squad_dataset(tokenizer, args.batch_size)
    # model.print_trainable_parameters()

    optimizer = AdamWeightDecay(
        params=model.trainable_params(), learning_rate=args.lr)

    model = ConvBertTrainNet(model)
    model = Model(network=model, loss_fn=None, optimizer=optimizer)
    model.build(ds, epoch=args.num_epochs)
    model.train(
        epoch=args.num_epochs,
        train_dataset=ds,
        callbacks=LossMonitor(per_print_times=1),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=2, type=int, help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--model_name_or_path",
        default=MODEL_PATH,
        type=str,
        help="YituTech/conv-bert-base",
    )
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument(
        "--lr", default=1e-4, type=float, help="Set 2e-5 for full-finetuning."
    )
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--model_save_dir", type=str,
                        default="convbert_lora_peft")
    args = parser.parse_args()
    main(args)
