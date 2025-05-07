import sys
from io import  StringIO
import logging
from logging.handlers import RotatingFileHandler
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough,RunnableLambda
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
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


# 初始化日志系统
def setup_logging():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器，每个日志文件最大10MB，保留3个备份
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()



def get_ollama_llm():
    try:
        client = Client(
            host='http://localhost:11434',
            headers={'x-some-header': 'some-value'}
        )
        logger.info("Ollama客户端初始化成功")
        
        def ollama_chat(messages):
            try:
                logger.debug(f"发送消息到Ollama: {messages}")
                response = client.chat(
                    model='deepseek-r1:1.5b',
                    messages=messages,
                    stream=False 
                )
                logger.debug("从Ollama收到响应")
                return response['message']['content']
            except Exception as e:
                logger.error(f"Ollama聊天错误: {str(e)}")
                return "抱歉，处理您的请求时出现问题"
        
        return ollama_chat
    except Exception as e:
        logger.error(f"初始化Ollama客户端失败: {str(e)}")
        raise

def get_retriever():
    try:
        embedding = OllamaEmbeddings(
            model="mxbai-embed-large",
            base_url="http://localhost:11434"  
        )
        persist_directory = './data_base/vector_db/chroma'
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        logger.info("向量检索器初始化成功")
        return vectordb.as_retriever()
    except Exception as e:
        logger.error(f"初始化检索器失败: {str(e)}")
        raise


def gen_response(chain, input, chat_history,use_knowledge=True):
    response = chain({
        "input": input,
        "chat_history": chat_history,
        "use_knowledge": use_knowledge
    })
    yield response

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

def process_uploaded_files(uploaded_files):
    """处理上传的文件并存入向量数据库"""
    try:
        docs = []
        os.makedirs("./temp_uploads", exist_ok=True)
        logger.info(f"开始处理 {len(uploaded_files)} 个上传文件")
        
        for file in uploaded_files:
            file_path = f"./temp_uploads/{file.name}"
            try:
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                loader = None
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file.name.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif file.name.endswith(".txt"):
                    loader = TextLoader(file_path)
                elif file.name.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(file_path)
                
                if loader:
                    docs.extend(loader.load())
                    logger.info(f"成功加载文件: {file.name}")
                    os.remove(file_path)  # 删除临时文件
                    logger.info(f"删除临时文件: {file_path}")
            except Exception as e:
                logger.error(f"处理文件 {file.name} 时出错: {str(e)}")
        
        if docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            
            embedding = OllamaEmbeddings(
                base_url="http://localhost:11434",
                model="mxbai-embed-large"
            )
            
            try:
                Chroma.from_documents(
                    documents=splits,
                    embedding=embedding,
                    persist_directory="./data_base/vector_db/chroma"
                )
                logger.info(f"成功存储 {len(splits)} 个文档块到向量数据库")
            except Exception as e:
                logger.error(f"存储到向量数据库失败: {str(e)}")
                raise
    except Exception as e:
        logger.error(f"处理上传文件时发生错误: {str(e)}")
        raise

def get_qa_history_chain():
    try:
        retriever = get_retriever()
        llm = get_ollama_llm()

        knowledge_system_prompt = (
            "你是一个问答任务的助手。请严格根据提供的上下文回答问题。\n"
            "上下文：\n{context}\n\n"
            "问题：{input}\n"
            "如果上下文不包含答案，请回答'我不知道'。"
        )
        
        general_system_prompt = (
            "你是一个智能助手，请直接回答用户的问题。不需要参考特定文档。"
        )
        
        def run_ollama_chain(input_data):
            try:
                use_knowledge = input_data.get("use_knowledge", True)
                logger.info(f"使用知识库: {use_knowledge}, 问题: {input_data['input']}")
                
                if use_knowledge:
                    # 使用知识库检索
                    retrieved_docs = retriever.get_relevant_documents(input_data["input"])
                    logger.debug(f"检索到 {len(retrieved_docs)} 个相关文档")
                    
                    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                    messages = [
                         {
                        "role": "system",
                        "content": knowledge_system_prompt.format(
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
                else:
                    # 不使用知识库
                    
                    messages = [
                        {
                            "role": "system",
                            "content": general_system_prompt.format(
                                context=input_data.get("context", ""),
                                input=input_data.get("input", "")
                        )
                        },
                        *[
                            {"role": "user" if msg[0] == "human" else "assistant", "content": msg[1]}
                            for msg in input_data.get("chat_history", [])
                        ],
                        {"role": "user", "content": input_data["input"]}
                    ]
                
                # 添加聊天历史
                response = llm(messages)
                logger.debug("生成回答成功")
                return response
            except Exception as e:
                logger.error(f"生成回答时出错: {str(e)}")
                return "抱歉，处理您的请求时出现问题"
        
        return run_ollama_chain
    except Exception as e:
        logger.error(f"初始化问答链失败: {str(e)}")
        raise

def main():
    try:
        st.markdown('### 🦜🔗 动手学大模型应用开发')
        with st.sidebar:
            # 添加知识库开关
            use_knowledge = st.toggle(
                "使用知识库回答",
                value=True,
                help="启用后会基于上传的文件内容回答，禁用则使用模型的一般知识"
            )
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
        
        if "qa_history_chain" not in st.session_state:
            st.session_state.qa_history_chain = get_qa_history_chain()
        
        messages = st.container(height=550)
        for message in st.session_state.messages:
            with messages.chat_message(message[0]):
                st.write(message[1])
        
        prompt = st.chat_input("Say something")
        if prompt:
            st.session_state.messages.append(("human", prompt))
            with messages.chat_message("human"):
                st.write(prompt)
            
            try:
                answer = gen_response(
                    chain=st.session_state.qa_history_chain,
                    input=prompt,
                    chat_history=st.session_state.messages,
                    use_knowledge=use_knowledge
                )
                with messages.chat_message("ai"):
                    output = st.write_stream(answer)
                st.session_state.messages.append(("ai", output))
            except Exception as e:
                logger.error(f"处理用户输入时出错: {str(e)}")
                st.error("处理您的请求时出现问题")
    except Exception as e:
        logger.critical(f"应用程序崩溃: {str(e)}")
        st.error("应用程序发生严重错误")

if __name__ == "__main__":
    main()
