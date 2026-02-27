import sqlite3
import logging
from dotenv import load_dotenv
from typing import Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableWithMessageHistory,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory

load_dotenv()


class Prompts:
    """提示词管理类

    管理系统所需的各类提示词模板，包括：
    - system_prompt: 系统角色和行为定义
    - greeting_prompt: 初始欢迎语
    - prompt_template: 对话模板格式
    """

    # 系统提示词
    system_prompt = """你是一个名叫Molly的医学专家，
            对于用户提问的医学相关问题，你需要按照给出的参考文献资料对问题进行回答。
            你的回答需要按照以下步骤：
                1. 分析用户问题、对话历史以及参考文献，判断参考资料的哪些内容可以解答用户的问题，并将这一过程进行说明。
                2. 如果参考文献可以解答用户的问题，则根据文献内容对问题进行解答。
                3. 如果参考文献不能解答用户问题，告诉用户信息不足，无法回答，建议用户寻求专业人士帮助，不要自行发挥。
            你的回答需要注意以下几点： 
                1. 保证你的回答是清晰的、明确的。如果你参考了参考资料，应该指出参考资料的标题等。
                2. 结合用户的对话历史，分析用户的问题意图。但不要复述问题。
                3. 回复用户时，使用对话的口吻，有礼貌地称呼用户为”您“，不要使用“用户”来称呼！
                4. 如果用户的问题与医学无关，判断用户的目的，并温柔地提示其回到医学话题。
            再次提醒：请严格遵守以上规则，当参考资料不足时，拒绝回答问题，不要自行发挥！"""

    # 欢迎提示词
    greeting_prompt = (
        "你好！我是Molly医疗精灵，专注解决你的医疗问题。请问你需要什么帮助？"
    )

    # 对话提示词模版
    prompt_template = """##用户问题：{input}
        ##本地知识库：{rag_results} 
        ##对话历史：{chat_history}"""


class Robot:
    def __init__(self, retriever=None) -> None:
        # 实例化提示词管理类
        self.prompts = Prompts()

        # 初始化大模型
        llm = ChatOpenAI(model="gpt-4o")

        # 构建提示词模板
        template = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts.system_prompt),
                ("human", self.prompts.prompt_template),
            ]
        )

        # 构建对话历史存储
        chat_hist = RunnableWithMessageHistory(
            template | llm,
            get_session_history=self.get_history,
            history_messages_key="chat_history",
        )

        # 构建检索器
        if retriever is None:
            retriever = RunnableLambda(lambda x: "No retriever provided")

        # 构建链
        self.chain = {
            "input": RunnablePassthrough(),
            "rag_results": retriever,
            "chat_history": RunnablePassthrough(),
        } | chat_hist

        logging.info("初始化完成")

    # 对话历史存储
    def get_history(self, session_id):
        # 如果传入的session_id不在数据库中，则说明是新的会话，需要新创建
        if session_id not in self.check_session_ids():
            logging.info(f"会话ID {session_id} 不存在，创建新会话")
            # 新创建会话，连接到chat_records.db数据库
            history = SQLChatMessageHistory(
                session_id=session_id, connection_string="sqlite:///chat_records.db"
            )
            # 将欢迎提示词传入到会话历史中
            history.add_message(AIMessage(self.prompts.greeting_prompt))
            logging.info(f"会话ID {session_id} 已创建，欢迎提示词已添加")
        else:
            # 如果会话ID存在，则直接返回会话历史
            history = SQLChatMessageHistory(
                session_id=session_id, connection_string="sqlite:///chat_records.db"
            )
            logging.info(f"会话ID {session_id} 存在，返回会话历史")
        return history

    # 检查会话ID是否存在,存在则返回session_id列表，不存在则返回空列表
    def check_session_ids(self):
        try:
            with sqlite3.connect("chat_records.db") as conn:
                # 创建游标
                cursur = conn.cursor()
                # 查询是否存在数据表
                cursur.execute(
                    "select count(*) from sqlite_master where type = 'table' and name = 'message_store'"
                )
                # 根据返回结果判断是否存在指定数据表
                # 如果上面的操作返回的是0，说明不存在指定的表，则直接返回空列表
                if cursur.fetchone()[0] == 0:
                    logging.info("数据表 message_store 不存在")
                    return []
                # 如果存在当前数据表，则查询所有的session_id并返回
                resp = cursur.execute("SELECT DISTINCT session_id FROM message_store")
                # 从查询结果中提取所有的session_id
                response = resp.fetchall()
                session_ids = [row[0] for row in response]
                logging.info("获取当前数据库中的会话ID列表")
                return session_ids
        except sqlite3.Error as e:
            logging.error(f"数据库查询错误: {e}")
            return []

    # 大模型返回内容1：全部加载完成之后输出
    def chat(self, input: str, session_id: str) -> str:
        # 这种结构是LangChain框架要求的格式，RunnableWithMessageHistory 需要使用这种特定格式来识别和管理会话历史
        # config 它包含了一个嵌套字典 "configurable"，其中存储了当前会话的 session_id
        config = {"configurable": {"session_id": session_id}}
        # 调用链的invoke方法，传入用户输入和会话配置
        response = self.chain.invoke(input, config=config)  # type: ignore

        # 将大模型生成的内容返回
        return response.content

    # 大模型返回内容2：流式输出
    def stream(self, input: str, session_id: str):
        if input and session_id:
            # 这种结构是LangChain框架要求的格式，RunnableWithMessageHistory 需要使用这种特定格式来识别和管理会话历史
            config = {"configurable": {"session_id": session_id}}
            # 调用链的stream方法，传入用户输入和会话配置
            response = self.chain.stream(input, config=config)  # type: ignore

            # 将生成器直接返回
            return response  # type: ignore
        else:
            logging.error("输入或会话ID为空，无法流式输出")
            return None
