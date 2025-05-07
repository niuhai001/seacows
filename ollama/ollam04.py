import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough,RunnableLambda
# import sys
# sys.path.append("zhipuai_embedding.py")  # 将父目录放入系统路径中
# from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.llms import SparkLLM
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime 
from ollama import Client
from langchain_community.embeddings import OllamaEmbeddings


# import asyncio
# from langchain.embeddings import AsyncEmbeddingWrapper

# tianji = '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'




def get_ollama_llm():
    client = Client(
    host='http://localhost:11434',
    headers={'x-some-header': 'some-value'}
    )
    def ollama_chat(messages):
        response = client.chat(
            model='deepseek-r1:1.5b',  # 或其他你安装的模型如 'mistral', 'qwen'等
            messages=messages,
            stream=False 
        )
        return response['message']['content']
    
    return ollama_chat



def get_retriever():
    # 定义 Embeddings
    embedding = OllamaEmbeddings(
        model="mxbai-embed-large",  # 推荐轻量级嵌入模型
        base_url="http://localhost:11434"  
    )
    # 向量数据库持久化路径
    persist_directory = './data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain():
    retriever = get_retriever()
    llm = get_ollama_llm()  # 使用新的Ollama客户端
    
    
    # 修改prompt模板以适应Ollama
    system_prompt = (
        "你是一个问答任务的助手。请使用检索到的上下文片段回答这个问题。\n"
        "上下文：\n{context}\n\n"
        "问题：{input}"
    )
    
    def run_ollama_chain(input_data):
        # 1. 先检索知识库
        retrieved_docs = retriever.get_relevant_documents(input_data["input"])
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 2. 构建带上下文的提示
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    context=context,
                    input=input_data["input"]
                )
            },
            *[
                {"role": "user" if msg[0] == "human" else "assistant", "content": msg[1]}
                for msg in input_data.get("chat_history", [])
            ],
            {"role": "user", "content": input_data["input"]}
        ]
        
        # 3. 调用LLM生成回答
        return llm(messages)
    
    return run_ollama_chain

def gen_response(chain, input, chat_history):
    response = chain({
        "input": input,
        "chat_history": chat_history
    })
    yield response


def process_uploaded_files(uploaded_files):
    """处理上传的文件并存入向量数据库"""
    docs = []
    
    # 创建临时目录存储文件
    os.makedirs("./temp_uploads", exist_ok=True)
   
    
    for file in uploaded_files:
        # 保存上传文件
        file_path = f"./temp_uploads/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # 根据文件类型选择加载器
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.name.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            continue
            
        docs.extend(loader.load())
        # os.remove(file_path)  # 清理临时文件
    if docs:
        embedding = OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")
        Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory="./data_base/vector_db/chroma"
        )
    if docs:
        # 文本分块处理
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        splits = text_splitter.split_documents(docs)
        batch_size = 64
    
        # 向量化存储
        embedding =  OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")
        Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory="./data_base/vector_db/chroma"
        )
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            Chroma.from_documents(
                documents=batch,
                embedding=embedding,
                persist_directory="./data_base/vector_db/chroma"
            )
    
        for doc in docs:
        # 添加文件来源信息
            doc.metadata.update({
                "source": os.path.basename(doc.metadata.get("source", "")),
                "processed_time": datetime.now().isoformat()
            })


# Streamlit 应用程序界面
def main():
    st.markdown('### 🦜🔗 动手学大模型应用开发')
    # 用于跟踪对话历史
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "上传知识库文件",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True
        )
        if uploaded_files and st.button("处理文件"):
            with st.spinner("正在处理文件..."):
                process_uploaded_files(uploaded_files)
                st.success("文件处理完成！")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链get_chat_qa_chain
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages:
            with messages.chat_message(message[0]):
                st.write(message[1])
    prompt = st.chat_input("Say something")
    if prompt:
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # st.session_state.messages.append({"role": "user", "text": prompt})
        with messages.chat_message("human"):
            st.write(prompt)
     
     
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        st.session_state.messages.append(("ai", output))


if __name__ == "__main__":
    main()