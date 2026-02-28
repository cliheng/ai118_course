from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

load_dotenv()
def get_history(session_id):
    """
    根据会话 ID 获取聊天历史记录对象。

    参数:
    session_id (str): 会话的唯一标识符，用于区分不同的聊天会话。

    返回:
    SQLChatMessageHistory: 一个 SQLChatMessageHistory 实例，用于管理该会话的聊天历史记录。
    """
    # 使用 create_engine 函数创建一个 SQLite 数据库引擎
    # 数据库文件名为 chat_history.db
    engine = create_engine("sqlite:///chat_history.db")
    # 创建 SQLChatMessageHistory 实例，将会话 ID 和数据库连接传递给它
    return SQLChatMessageHistory(session_id, connection=engine.connect())

if __name__ == "__main__":
    # 设置 OpenAI API 密钥，通过环境变量的方式传递给 OpenAI 客户端
    os.environ["OPENAI_API_KEY"] = "fk233499-IV2MwP4aahyfLWf1E5pWqsOW5fchwyNP"
    # 设置 OpenAI API 的基础地址，指定 API 请求的目标地址
    os.environ["OPENAI_API_BASE"] = "https://oa.api2d.net"

    # 创建 ChatOpenAI 实例，指定使用的模型为 gpt-3.5-turbo
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # 创建 RunnableWithMessageHistory 实例，用于处理带有聊天历史的对话
    llm_hist = RunnableWithMessageHistory(
        runnable=llm,  # 可运行的对象，这里是 ChatOpenAI 实例
        llm=llm,  # LLM 模型实例
        get_session_history=get_history  # 获取聊天历史记录的函数
    )
    
    # 配置会话信息，指定会话 ID 为 abc123
    config = {'configurable': {'session_id': 'abc123'}}
    
    # 调用 llm_hist 实例，传入人类用户的消息和配置信息
    # 发送的消息内容为 "你好！"
    resp = llm_hist.invoke(
        HumanMessage(content="你好！"),
        config=config
    )
    # 打印 LLM 模型返回的响应结果
    print(resp)
