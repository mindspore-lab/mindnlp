# LLaMA 

## Setup

In a conda env with mindspore available, run:
```
pip install -r requirements.txt
```

## Download

Download checkpoint files and tokenizers. Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

## Convert weight

Convert weights from Pytorch to MindSpore.

```
python convert.py --ckpt_path $TARGET_FOLDER/model_size
```

## Inference

The provided `example.py` can be run on a single or multi-card node with `mpirun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```
mpirun -n MP python example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

Different models require different MP values (at least 24GB per card):

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |

## Reference

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```