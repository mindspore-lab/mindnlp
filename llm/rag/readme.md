### RAG demo by MindNLP+Langchain

#### Install dependencies

```
pip install mindnlp langchain langchain-community faiss-cpu
```

### Download knowledge file

```
wget https://raw.githubusercontent.com/limchiahooi/nlp-chinese/master/%E8%A5%BF%E6%B8%B8%E8%AE%B0.txt -O xiyouji.txt

```

### Run RAG Demo

```
streamlit run startup.py xiyouji.txt 
```