import pandas as pd
from mindnlp.transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from tqdm import tqdm
import argparse

MODEL_CONFIGS = {
    "bart": {
        "model_name": "ModelTC/bart-base-qnli",
        "tokenizer_name": "facebook/bart-base"
    },
    "bert": {
        "model_name": "Li/bert-base-uncased-qnli",
        "tokenizer_name": "google-bert/bert-base-uncased"
    },
    "roberta": {
        "model_name": "howey/roberta-large-qnli",
        "tokenizer_name": "FacebookAI/roberta-large"
    },
    "xlm-roberta": {
        "model_name": "tmnam20/xlm-roberta-large-qnli-1",
        "tokenizer_name": "FacebookAI/xlm-roberta-large"
    },
    "gpt2": {
        "model_name": "tanganke/gpt2_qnli",
        "tokenizer_name": "openai-community/gpt2"
    },
    "t5": {
        "model_name": "lightsout19/t5-small-qnli",
        "tokenizer_name": "google-t5/t5-small"
    },
    "distilbert": {
        "model_name": "anirudh21/distilbert-base-uncased-finetuned-qnli",
        "tokenizer_name": "distilbert/distilbert-base-uncased"
    },
    "llama": {
        "model_name": "Cheng98/llama-160m-qnli",
        "tokenizer_name": "JackFram/llama-160m"
    },
    "albert": {
        "model_name": "orafandina/albert-base-v2-finetuned-qnli",
        "tokenizer_name": "albert/albert-base-v2"
    },
    "opt": {
        "model_name": "utahnlp/qnli_facebook_opt-125m_seed-1",
        "tokenizer_name": "facebook/opt-125m"
    },
    "llama": {
        "model_name": "Cheng98/llama-160m-qnli",
        "tokenizer_name": "JackFram/llama-160m"
    },
}

def get_model_and_tokenizer(model_type):
    """获取指定类型的模型和分词器"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    config = MODEL_CONFIGS[model_type]
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)
    
    return model, tokenizer

def predict_qnli(model, tokenizer, question, sentence):
    """预测QNLI任务"""
    inputs = tokenizer(question, sentence, return_tensors="ms", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    return logits.argmax(axis=1).asnumpy()[0]

def evaluate_model(model_type, data_path):
    """评估模型在QNLI数据集上的表现"""
    print(f"正在评估模型: {model_type}")
    
    model, tokenizer = get_model_and_tokenizer(model_type)
    print(f"模型类型: {model.config.model_type}")
    
    df = pd.read_csv(data_path, sep='\t', header=0, names=['idx', 'question', 'sentence', 'label'], on_bad_lines='skip')
    df = df.dropna(subset=['label'])
    
    label_map = {'entailment': 0, 'not_entailment': 1}
    valid_data = df[df['label'].isin(label_map.keys())]
    
    questions = valid_data['question'].tolist()
    sentences = valid_data['sentence'].tolist()
    labels = [label_map[label] for label in valid_data['label']]
    
    predict_true = 0
    for question, sentence, true_label in tqdm(zip(questions, sentences, labels), 
                                             total=len(questions), 
                                             desc="预测进度"):
        pred_label = predict_qnli(model, tokenizer, question, sentence)
        if pred_label == true_label:
            predict_true += 1
    
    accuracy = float(predict_true / len(questions) * 100)
    print(f"测试集总样本数: {len(questions)}")
    print(f"预测正确的数量: {predict_true}")
    print(f"准确率为: {accuracy:.2f}%")
    
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QNLI任务评估脚本')
    parser.add_argument('--model', type=str, required=True, 
                       choices=list(MODEL_CONFIGS.keys()),
                       help='要评估的模型类型')
    parser.add_argument('--data', type=str, default='./QNLI/dev.tsv',
                       help='数据集路径')
    
    args = parser.parse_args()
    evaluate_model(args.model, args.data)