# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import argparse
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

import mindhf
from embedding import EmbeddingsFunAdapter
from text import TextLoader
from threading import Thread

import mindspore
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

def load_knowledge_base(file_name):
    print(f"正在加载知识库文件: {file_name}")
    loader = TextLoader(file_name)
    texts = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=256, chunk_overlap=0)
    split_docs = text_splitter.split_text(texts)
    print(f"文档已切分为 {len(split_docs)} 个片段")

    embeddings = EmbeddingsFunAdapter("Qwen/Qwen3-Embedding-0.6B")
    faiss = FAISS.from_texts(split_docs, embeddings)
    print("FAISS 向量数据库构建完成。")
    return faiss


def load_model_and_tokenizer():
    print("正在加载模型")
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', use_fast=False, mirror='modelscope', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', ms_dtype=mindspore.bfloat16,mirror='modelscope', device_map=0)

    
    print("模型加载完成。")
    return tokenizer, model


def retrieve_knowledge(faiss, query):
    docs = faiss.similarity_search(query, k=1)  
    return docs[0].page_content

def generate_answer(tokenizer, model, query, knowledge=None):
    if knowledge:
        input_text = knowledge + "\n\n" + query
    else:
        input_text = query

    messages = [
        {"role": "user", "content": input_text}
    ]

    # 使用 tokenizer.apply_chat_template 构建输入
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"⚠️  apply_chat_template 失败，使用手动拼接: {e}")
        prompt = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)

    # 创建 streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,           # 跳过输入部分
        skip_special_tokens=True    # 不输出特殊 token
    )

    # 启动生成线程
    def generate():
        model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.001,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    thread = Thread(target=generate)
    thread.start()

    # 实时输出生成的文本
    print("回答: ", end="", flush=True)
    generated_text = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        generated_text += new_text
    print()  # 换行

    return generated_text.strip()



def rag_pipeline(faiss, tokenizer, model, query, use_rag=True):
    if use_rag:
        knowledge = retrieve_knowledge(faiss, query)
        answer = generate_answer(tokenizer, model, query, knowledge)
        return answer, knowledge
    else:
        answer = generate_answer(tokenizer, model, query, "")
        return answer, ""


def main():
    parser = argparse.ArgumentParser(description="RAG Demo - Command Line Version")
    parser.add_argument("filename", help="知识库文本文件路径")
    args = parser.parse_args()

    # 加载知识库和模型
    faiss_db = load_knowledge_base(args.filename)
    tokenizer, model = load_model_and_tokenizer()

    print("\n" + "="*60)
    print("RAG系统已准备就绪！")
    print("输入 'quit' 或 'exit' 退出程序。")
    print("="*60)

    while True:
        try:
            # 获取用户输入
            query = input("\n请输入您的问题: ").strip()
            if query.lower() in ['quit', 'exit', 'bye']:
                print("再见！")
                break
            if not query:
                print("问题不能为空，请重新输入。")
                continue

            # 是否启用 RAG
            use_rag_input = input("是否启用检索增强 (RAG)? [Y/n]: ").strip().lower()
            use_rag = use_rag_input not in ['n', 'no', 'N', 'NO']

            # RAG 流程
            if use_rag:
                print("正在检索知识库...")
                knowledge = retrieve_knowledge(faiss_db, query)
                print(f"检索到的知识:\n{knowledge}")
                # print("生成中: ", end="", flush=True)
                answer = generate_answer(tokenizer, model, query, knowledge)
            else:
                print("直接生成回答（无检索）...")
                # print("生成中: ", end="", flush=True)
                answer = generate_answer(tokenizer, model, query)

        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break

            
if __name__ == "__main__":
    main()