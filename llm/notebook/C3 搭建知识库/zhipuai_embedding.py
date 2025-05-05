import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
# import sys
# sys.path.append("zhipuai_embedding.py")  # 将父目录放入系统路径中
# from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import SparkLLM
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime 
# import asyncio
# from langchain.embeddings import AsyncEmbeddingWrapper

# tianji = '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'


ZHIPUAI_API_KEY = "1a910c1a587e42d489d031fec8e519bb.TLsuC8OkBq0xAtMI"
IFLYTEK_SPARK_APP_ID = 'c31c86a6'
IFLYTEK_SPARK_API_KEY = '982b899ec4da15cbe2156af77a536430'
IFLYTEK_SPARK_API_SECRET = 'YjdkNDQyMjllZDRjM2U0NDJjY2E4NGM5'

def gen_spark_params(model):
    '''
    构造星火模型请求参数
    '''

    spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"
    model_params_dict = {
        # v1.5 版本
        "v1.5": {
            "domain": "general",  # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v1.1")  # 云端环境的服务地址
        },
        # v2.0 版本
        "v2.0": {
            "domain": "generalv2",  # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v2.1")  # 云端环境的服务地址
        },
        # v3.0 版本
        "v3.0": {
            "domain": "generalv3",  # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.1")  # 云端环境的服务地址
        },
        # v3.5 版本
        "v3.5": {
            "domain": "generalv3.5",  # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.5")  # 云端环境的服务地址
        },
        # v4.0 版本
        "v4.0": {
            "domain": "4.0Ultra",  # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v4.0")  # 云端环境的服务地址
        }
    }
    return model_params_dict[model]




def get_retriever():
    # 定义 Embeddings
    api_key = ZHIPUAI_API_KEY
    embedding = ZhipuAIEmbeddings(
        api_key= api_key,
        model="embedding-2"
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

    llm = SparkLLM(
        spark_api_url=gen_spark_params(model="v4.0")["spark_url"],
        spark_app_id=IFLYTEK_SPARK_APP_ID,
        spark_api_key=IFLYTEK_SPARK_API_KEY,
        spark_api_secret=IFLYTEK_SPARK_API_SECRET,
        spark_llm_domain=gen_spark_params(model="v4.0")["domain"],
        streaming=False,
    )
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
        "请用中文回答。"
        "回到条数不能少于5条"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        # "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs,
        ).assign(answer=qa_chain)
    return qa_history_chain


def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]


async def async_embed():
    embedder = AsyncEmbeddingWrapper(ZhipuAIEmbeddings(
        api_key=ZHIPUAI_API_KEY,
        model="embedding-2"
    ))
    return await embedder.aembed_documents(texts)


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
        os.remove(file_path)  # 清理临时文件
    if docs:
        embedding = ZhipuAIEmbeddings(api_key=ZHIPUAI_API_KEY, model="embedding-2")
        Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory="./data_base/vector_db/chroma"
        )
    if docs:
        # 文本分块处理
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len,
            is_separator_regex=False,
        )
        splits = text_splitter.split_documents(docs)
        batch_size = 64
    
        # 向量化存储
        embedding = ZhipuAIEmbeddings(api_key=ZHIPUAI_API_KEY, model="embedding-2")
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