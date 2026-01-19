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
"""
Web RAG demo quick start
"""
import argparse
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st

import mindspore
from mindnlp.transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from embedding import EmbeddingsFunAdapter
from text import TextLoader


@st.cache_resource
def load_knowledge_base(file_name):
    loader = TextLoader(file_name)
    texts = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=256, chunk_overlap=0)
    split_docs = text_splitter.split_text(texts)
    print(len(split_docs))
    embeddings = EmbeddingsFunAdapter("AI-ModelScope/m3e-base", mirror='modelscope')
    faiss = FAISS.from_texts(split_docs, embeddings)
    return faiss


@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForSeq2SeqLM.from_pretrained("ZhipuAI/ChatGLM-6B", mirror='modelscope', ms_dtype=mindspore.float16)
    model.set_train(False)
    tokenizer = AutoTokenizer.from_pretrained("ZhipuAI/ChatGLM-6B", mirror='modelscope')
    return tokenizer, model


def retrieve_knowledge(faiss, query):
    docs = faiss.similarity_search(query)
    return docs[0].page_content


def generate_answer(tokenizer, model, query, knowledge):
    input_text = knowledge + query
    response = model.chat(tokenizer, input_text, temperature=0.001)
    answer = response[0]
    return answer


def rag_pipeline(faiss, tokenizer, model, query, use_rag):
    if use_rag:
        knowledge = retrieve_knowledge(faiss, query)
        answer = generate_answer(tokenizer, model, query, knowledge)
        return answer, knowledge
    else:
        answer = generate_answer(tokenizer, model, query, "")
        return answer, ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="args for RAG Demo")
    parser.add_argument('filename', help="knowledge base file")
    args = parser.parse_args()

    st.title("RAG Demo")
    faiss = load_knowledge_base(args.filename)
    tokenizer, model = load_model_and_tokenizer()

    if 'answer' not in st.session_state:
        st.session_state['answer'] = ""

    if 'knowledge' not in st.session_state:
        st.session_state['knowledge'] = ""

    with st.form(key='chat_form', clear_on_submit=True):
        query = st.text_input("输入您的问题：", "")
        use_rag = st.checkbox("启用检索增强生成 (RAG)", value=True)
        submitted = st.form_submit_button("发送")

    if submitted and query:
        answer, knowledge = rag_pipeline(faiss, tokenizer, model, query, use_rag)
        st.session_state['answer'] = answer
        st.session_state['knowledge'] = knowledge

    elif submitted:
        st.warning("请输入一个问题。")

    with st.subheader("用户输入"):
        st.text_area("User Query", query, height=50)

    # 左右布局
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("回答")
        st.text_area("Assistant Answer", st.session_state['answer'], height=300)

    with right_column:
        st.subheader("知识检索")
        st.text_area("Knowledge", st.session_state['knowledge'], height=300)

    # 重新运行以刷新对话历史
    if submitted:
        st.rerun()
