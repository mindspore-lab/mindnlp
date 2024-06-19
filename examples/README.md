# MindNLP Examples

MindNLP currently supports a variety of different NLP tasks and offers a wide range of state-of-the-art open-source models. We provide them in the form of examples.

## Supported Tasks in MindNLP 💡

MindNLP is a versatile repository that supports a variety of natural language processing tasks. It offers a wide array of state-of-the-art models for these tasks. Here's a brief overview:

### Classification 📊

MindNLP supports text classification tasks, including sentiment analysis, document classification, and more. You can quickly classify text into predefined categories or analyze sentiment.

| Task               | Model   | Dataset  | Example |
|--------------------|---------|----------|---------|
| Sentiment analysis | BERT    | Emotect  | [Notebook](./classification/bert_emotect_finetune.ipynb) |
|                    | GPT     | IMDB     | [Notebook](./classification/gpt_imdb_finetune.ipynb) |
|                    | Bi-LSTM | IMDB     | [Notebook](./classification/bilstm_imdb_concise.ipynb) |
| Chinese news       | NeZha   | THUCNews | [Notebook](./classification/nezha_classification.ipynb) |

### Language Model 🧠

MindNLP provides access to cutting-edge language models, which can be used for tasks like text generation, text completion, and text classification. These models are highly capable of understanding and generating human-like text.

| Model   | Dataset  | Example |
|---------|----------|---------|
| FastText  | AGNews | [Script](./language_model/fasttext.py) |

### Machine Translation 🌐

MindNLP supports machine translation, allowing you to translate text from one language to another. It covers a wide range of language pairs and ensures accurate translations.

| Model   | Dataset  | Example |
|---------|----------|---------|
| Seq2seq(GRU)  | Multi30k | [Notebook](./machine_translation/mindspore_sequence_to_sequence.ipynb) |

### Question Answer❓

You can build question answering systems using MindNLP. Given a context and a question, these models can extract answers directly from the provided text.

| Model   | Dataset  | Example |
|---------|----------|---------|
| Bidaf  | Squad1 | [Notebook](./question_answer/bidaf_squad_concise.ipynb) |

### Sequence Labeling 🏷️

For tasks like named entity recognition (NER) and part-of-speech tagging, MindNLP offers sequence labeling models. These models can identify and label entities or segments within a text.

| Task               | Model   | Dataset  | Example |
|--------------------|---------|----------|---------|
| Named Entity Recognation | Bi-LSTM+CRF | Coll2003 | [Notebook](./sequence_labeling/LSTM-CRF.ipynb) |
|  | BERT+Bi-LSTM+CRF | Coll2003 | [Notebook](./sequence_labeling/Bert-LSTM-CRF.ipynb) |

### Text Generation 📝

MindNLP includes models for text generation, which can create new text based on provided prompts, generate creative content, or produce concise summaries of long documents or articles.

| Task               | Model   | Dataset  | Example |
|--------------------|---------|----------|---------|
| Named Entity Recognation | GPT2 | NLPCC2017 | [Notebook](./text_generation/gpt2_summarization.ipynb) |

<!-- ### Language Understanding 🧐

In addition to the mentioned tasks, MindNLP supports various other language understanding tasks, including text entailment, paraphrasing, and more. -->
