+++
date = '2024-12-06T11:58:35+08:00'
draft = false
title = 'Langchain尝试'

+++

# LangChain浅尝试

LangChain 是一个[应用框架](https://zh.wikipedia.org/wiki/应用框架)，旨在简化使用[大型语言模型](https://zh.wikipedia.org/wiki/大型语言模型)的应用程序。作为一个语言模型集成框架，LangChain 的用例与一般[语言模型](https://zh.wikipedia.org/wiki/语言模型)的用例有很大的重叠。 重叠范围包括文档分析和总结摘要, 代码分析和[聊天机器人](https://zh.wikipedia.org/wiki/聊天機器人)。[[1\]](https://zh.wikipedia.org/zh-cn/LangChain#cite_note-1)LangChain提供了一个标准接口，用于将不同的[语言模型](https://zh.wikipedia.org/wiki/语言模型)（LLM）连接在一起，以及与其他工具和数据源的集成。LangChain还为常见应用程序提供端到端链，如[聊天机器人](https://zh.wikipedia.org/wiki/聊天機器人)、文档分析和代码生成。 LangChain是由Harrison Chase于2022年10月推出的[开源软件](https://zh.wikipedia.org/wiki/开源软件)项目。它已成为LLM开发中最受欢迎的框架之一。

## RAG工作模式

**Retrive**: give user a input , relevant splits are retrived from storage using  a Retriever

**Generate**: A [ChatModel](https://python.langchain.com/docs/concepts/chat_models/) / [LLM](https://python.langchain.com/docs/concepts/text_llms/) produces an answer using a prompt that includes both the question with the retrieved data

<img src="https://python.langchain.com/assets/images/rag_retrieval_generation-1046a4668d6bb08786ef73c56d4f228a.png" alt="retrieval_diagram" style="zoom:33%;" />

## 语句向量化

数据向量化可分为以下几个步骤：

加载数据集

划分数据

存储数据（向量数据库）

<img src="https://python.langchain.com/assets/images/rag_indexing-8160f90a90a33253d0154659cf7d453f.png" alt="index_diagram" style="zoom:33%;" />

## 构建RAG(TEST)

1. **A chat model**

   ChatGPT

   ```bash
   pip install -qU langchain-openai
   ```

   ```python
   import getpass
   import os
   
   os.environ["OPENAI_API_KEY"] = "your_openai_key"
   
   from langchain_openai import ChatOpenAI
   
   llm = ChatOpenAI(model="gpt-3-turbo")
   ```

   本地llama

   ```python
   from langchain_ollama import OllamaEmbeddings
   from langchain_ollama import ChatOllama
   embeddings = OllamaEmbeddings(
       model="wangshenzhi/llama3-8b-chinese-chat-ollama-q4",
   )
   llm = ChatOllama(model="wangshenzhi/llama3-8b-chinese-chat-ollama-q4")
   ```

2. **A embedding model**

   ```
   pip install -qU langchain-openai
   ```

   ```python
   import getpass
   
   os.environ["OPENAI_API_KEY"] = "your_openai_key"
   
   from langchain_openai import OpenAIEmbeddings
   
   embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
   ```

3. **A vector store**

   这里以[**PostgreSQL**](https://github.com/pgvector/pgvector)为向量数据库

   ```bash
   pip install -qU langchain-postgres
   ```

   ```python
   from langchain_postgres import PGVector
   
   vector_store = PGVector(
       embedding=embeddings,
       collection_name="my_docs",
       connection="postgresql+psycopg://...",
   )
   ```

4. **解析文本**

   ```python
   from langchain import hub
   from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
   from langchain_core.documents import Document
   from langchain_text_splitters import RecursiveCharacterTextSplitter
   from langgraph.graph import START, StateGraph
   loader = PyPDFLoader('./PCTA报考.pdf')
   pages = []
   async for page in loader.alazy_load():
       pages.append(page)
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
   all_splits = text_splitter.split_documents(pages)
   _ = vector_store.add_documents(documents=all_splits)
   ```

5. **数据相关性**

   ```python
   prompt = hub.pull("rlm/rag-prompt")
   # Define state for application
   class State(TypedDict):
       question: str
       context: List[Document]
       answer: str
   
   
   # Define application steps
   def retrieve(state: State):
       retrieved_docs = vector_store.similarity_search(state["question"])
       return {"context": retrieved_docs}
   
   
   def generate(state: State):
       docs_content = "\n\n".join(doc.page_content for doc in state["context"])
       messages = prompt.invoke({"question": state["question"], "context": docs_content})
       response = llm.invoke(messages)
       return {"answer": response.content}
   
   
   # Compile application and test
   graph_builder = StateGraph(State).add_sequence([retrieve, generate])
   graph_builder.add_edge(START, "retrieve")
   graph = graph_builder.compile()
   ```

## 效果

```python
response = graph.invoke({"question": "我要报名PCTA考试，我要联系哪个老师。"})
print(response["answer"])
```

```
要报名PCTA考试，你需要联系叶老师。学校已经是PingCAP的教育合作伙伴，并且有资格参与该项的学习和考试。具体操作如下：首先登录PingCAP的学习中心，注册账号并选择常州工学院，然后进入课程“TiDB数据库核心原理与架构 [TiDB v6]”进行学习。完成课程后，你就可以报名参加PCTA考试了。如果你准备好参加考试，请联系叶老师进行确认和报名流程。
```

