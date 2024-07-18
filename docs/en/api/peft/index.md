# PEFT (Prompt Engineering with Fine-Tuning)

MindNLP's PEFT (Prompt Engineering with Fine-Tuning) is a methodology for fine-tuning natural language models using prompts. It allows users to tailor their models for specific tasks or domains by providing customized prompts, enabling better performance on downstream tasks.

## Introduction

PEFT leverages the power of prompts, which are short natural language descriptions of the task, to guide the model's understanding and behavior. By fine-tuning with prompts, users can steer the model's attention and reasoning towards relevant information, improving its performance on targeted tasks.

## Supported PEFT Algorithms

| Algorithm        | Description                                                  |
|------------------|--------------------------------------------------------------|
| [AdaLoRA](./tuners/adalora.md)          | Adaptable Prompting with Learned Rationales (AdaLoRA)        |
| [Adaption_Prompt](./tuners/adaption_prompt.md)  | Adaptation Prompting                                          |
| [IA3](./tuners/ia3.md)              | Iterative Alignments for Adaptable Alignment and Prompting (IA3) |
| [LoKr](./tuners/lokr.md)             | Large-scale k-shot Knowledge Representation (LoKr)           |
| [LoRA](./tuners/lora.md)             | Learnable Redirection of Attention (LoRA)                    |
| [Prompt Tuning](./tuners/prompt_tuning.md)     | Prompt Tuning fine-tunes models by optimizing the prompts used during fine-tuning.              |

Each algorithm offers unique approaches to prompt engineering and fine-tuning, allowing users to adapt models to diverse tasks and domains effectively.

