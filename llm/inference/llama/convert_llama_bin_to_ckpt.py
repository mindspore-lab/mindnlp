import os
import torch
import mindspore
import argparse


def combine_bin_files_to_ckpt(ckpt_path, save_path):
    bin_files = [file for file in os.listdir(ckpt_path) if file.endswith('.bin')]
    bin_files.sort()  # Sort the files to ensure consistent order
    print("load the following files:", bin_files)
    ms_ckpt = []
    for bin_file in bin_files:
        bin_file_path = os.path.join(ckpt_path, bin_file)
        state_dict = torch.load(bin_file_path)
        for k, v in state_dict.items():
            if "model.embed_tokens.weight" in k:
                k = "model.embed_tokens.embedding_table"
            ms_ckpt.append({'name': k, 'data': mindspore.Tensor(v.float().numpy())})
            print(k)
    mindspore.save_checkpoint(ms_ckpt, save_path)

    print([ckpt['name'] for ckpt in ms_ckpt])
    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/cjl/code/chat/collections/models/Llama-2-7b-hf")
    parser.add_argument("--save_path", type=str, default="/home/cjl/code/chat/collections/mindnlp-models/llama-2-7b-hf/llama.ckpt")
    args = parser.parse_args()

    combine_bin_files_to_ckpt(args.model_name_or_path, args.save_path)