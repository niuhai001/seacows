import streamlit as st
# from langchain_openai import ChatOpenAI
from langchain_community.llms import SparkLLM
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
sys.path.append("notebook/C3 搭建知识库") # 将父目录放入系统路径中
# from zhipuai_embedding import ZhipuAIEmbeddings
from zhipuai import ZhipuAI

# from zhipuai import ZhipuAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

from langchain.embeddings import OpenAIEmbeddings
#质谱
ZHIPUAI_API_KEY = "1a910c1a587e42d489d031fec8e519bb.TLsuC8OkBq0xAtMI"
# 星火模型配置
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
    embedding = ZhipuAI(api_key=api_key)
    embeddings = embedding.as_embeddings.create(
        model="embedding-2"
    )

    vectorstore = Chroma.from_documents(
        # documents=docs,
        embedding=embeddings,
        persist_directory="./data_base/vector_db/chroma"
    )
    # 向量数据库持久化路径
    # persist_directory = './data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=vectorstore,
        embedding_function=embeddings
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
        "请使用简洁的话语回答用户。"
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

# Streamlit 应用程序界面
def main():
    st.markdown('### 🦜🔗 动手学大模型应用开发')

    # 用于跟踪对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages:
            with messages.chat_message(message[0]):
                st.write(message[1])
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
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