import mindspore
from mindnlp.transformers import BioGptForSequenceClassification, BioGptTokenizer
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from tqdm import tqdm

# 加载BioGPT模型
model_name = 'microsoft/biogpt'
tokenizer = BioGptTokenizer.from_pretrained(model_name)
model = BioGptForSequenceClassification.from_pretrained(model_name)

# 定义标签映射
label_map = {0: 'no', 1: 'yes', 2: 'maybe'}

# 推理函数
def infer(question, context):
    inputs = tokenizer(question + ' ' + ' '.join(context), return_tensors='ms', truncation=True, padding=True, max_length=1024)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = mindspore.ops.Argmax(axis=1)(logits).asnumpy()[0]
    predicted_label = label_map[predicted_class]

    # 打印推理结果
    print(f"Question: {question}")
    print(f"Context: {' '.join(context)}")
    print(f"Predicted Label: {predicted_label}")
    print("="*50)

    return predicted_label

# 评估函数
def evaluate(evaluation_set):
    true_labels = []
    predicted_labels = []

    for item in tqdm(evaluation_set, desc="Evaluating", unit="item"):
        question = item['question']
        context = item['context']
        true_label = item['final_decision']

        predicted_label = infer(question, context)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

# 加载数据集
ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial")

# 准备评估数据集
evaluation_set = []
for item in ds['train']:
    evaluation_set.append({
        "question": item['question'],
        "context": item['context'],
        "final_decision": item['final_decision']
    })

# 进行评估
accuracy = evaluate(evaluation_set)
print(f'Accuracy: {accuracy * 100:.2f}%')
