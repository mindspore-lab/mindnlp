""" Finetuning the library models for sequence classification ."""

from __future__ import absolute_import, division, print_function
import time
import os
import numpy as np
import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore.dataset import text
from mindspore import nn, Tensor, load_checkpoint, save_checkpoint
from mindspore.dataset import RandomSampler,  DistributedSampler, NumpySlicesDataset, SequentialSampler
from mindspore.communication.management import init, get_group_size

from mindnlp.models.mobilebert.mobilebert import MobileBertForSequenceClassification
from mindnlp.models.mobilebert.mobilebert_config import MobileBertConfig

from mindnlp.transforms.tokenizers.bert_tokenizer import BertTokenizer

from metrics.glue_compute_metrics import compute_metrics
from processors import glue_output_modes as output_modes
from processors import glue_processors as processors
from processors import glue_convert_examples_to_features as convert_examples_to_features
from processors import collate_fn
from tools.common import init_logger, logger
from callback.progressbar import ProgressBar
from tools.finetuning_argparse import get_argparse


MODEL_CLASSES = {
    "mobilebert": (MobileBertConfig, MobileBertForSequenceClassification, BertTokenizer),
}

def train(args, train_dataset, model):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler()
    train_dataloader =NumpySlicesDataset(train_dataset, sampler=train_sampler)
    train_dataloader = train_dataloader.batch(args.train_batch_size)
    if args.max_steps > 0:
        num_training_steps = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(num_training_steps * args.warmup_proportion)
    # Prepare optimizer (linear warmup and decay)
    optimizer_grouped_parameters = [
        {'params': list(filter(lambda x: 'bias' not in x.name and 'LayerNorm.weight' not in x.name, model.trainable_params())),
         'weight_decay': args.weight_decay},
        {'params': list(filter(lambda x: 'bias' in x.name or 'LayerNorm.weight' in x.name, model.trainable_params())), 'weight_decay': 0.0}
    ]
    optimizer = nn.AdamWeightDecay(optimizer_grouped_parameters, learning_rate=args.learning_rate, eps=args.adam_epsilon)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset[0]))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        def forward_fn(data, label):
            logits = model(**data)[1]
            loss = loss_fn(logits.view(-1, 2), label.view(-1))
            return loss, logits

        # Get gradient function
        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        # Define function of one-step training
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            loss = ops.depend(loss, optimizer(ops.clip_by_global_norm(grads, args.max_grad_norm)))
            return loss
        model.set_train()
        for batch, (all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels ) in enumerate(train_dataloader):
            inputs = {'input_ids': all_input_ids,
                      'attention_mask': all_attention_mask,
                      'labels': Tensor(all_labels, mindspore.int32)}
            inputs['token_type_ids'] = all_token_type_ids
            loss = train_step(inputs, Tensor(all_labels, mindspore.int32))
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss
            if (batch + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step//10))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                save_checkpoint(model, os.path.join(output_dir, 'mobilebert-uncased.ckpt'))
                logger.info("Saving model checkpoint to %s", output_dir)
            pbar(batch, {'loss': loss.asnumpy()})
        print(" ")
    return global_step, tr_loss / global_step, inputs


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, data_type='dev')
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * args.n_gpu
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler()
        eval_dataloader = NumpySlicesDataset(eval_dataset, sampler=eval_sampler)
        eval_dataloader = eval_dataloader.batch(args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        model.set_train(False)
        model.set_grad(False)
        for batch, (all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels ) in enumerate(eval_dataloader):
            inputs = {'input_ids': all_input_ids,
                      'attention_mask': all_attention_mask,
                      'labels': Tensor(all_labels, mindspore.int32)}
            inputs['token_type_ids'] = all_token_type_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss
            nb_eval_steps += 1
            if preds is None:
                preds = logits
                out_label_ids = inputs['labels']
            else:
                preds = np.append(preds, logits.asnumpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].asnumpy(), axis=0)
            pbar(batch)
        print(' ')
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return results


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    """load_and_cache_examples"""
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from dataset file
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if task in ['mnli', 'mnli-mm'] and 'roberta' in args.model_type:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]

    if data_type == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif data_type == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.max_seq_length,
                                            output_mode=output_mode)

    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    all_attention_mask = [f.attention_mask for f in features]
    all_token_type_ids = [f.token_type_ids for f in features]
    all_lens = [f.input_len for f in features]
    if output_mode == "classification":
        all_labels = [f.label for f in features]
    elif output_mode == "regression":
        all_labels = [f.label for f in features]
    dataset = ((all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels))
    return dataset


def main():
    args = get_argparse()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    args.n_gpu = 1
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    print(args)
    vocab = text.Vocab.from_file(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path+"/vocab.txt")
    tokenizer = tokenizer_class(
        vocab,
        lower_case=args.do_lower_case,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss, inputs = train(args, train_dataset, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        save_checkpoint(model, os.path.join(args.output_dir, 'mobilebert-uncased.ckpt'))

    if args.do_eval:
        print(evaluate(args, model, tokenizer))


if __name__ == "__main__":
    main()
