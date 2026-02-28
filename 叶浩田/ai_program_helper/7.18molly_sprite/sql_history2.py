from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableWithMessageHistory,RunnableLambda,RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
import sqlite3

class Prompts:
    # 系统提示词
    system_prompt = """你是一个名叫Molly的医学专家,
    对于用户提问的医学相关问题，你需要按照给出的参考文献资料对问题进行回答。
    你的回答需要按以下步骤：
        1. 分析用户问题、对话历史以及参考文献，判断参考资料的哪些内容可以解答用户的问题，并将这一过程进行说明。
        2. 如果参考文献可以解答用户的问题，则根据文献内容对问题进行解答。
        3. 如果参考文献不能解答用户的问题，告诉用户信息不足，无法回答，建议用户寻求专业人士帮助，不要自行发挥。
    你的回答需要注意以下几点：
        1. 保证你的回答是清晰的、明确的。如果你参考了参考资料，应该指出参考资料的标题等。
        2. 结合用户的对话历史，分析用户的问题意图。但不要复述问题。
        2. 回复用户时，使用对话的口吻，有礼貌地称呼用户为"您"，不要使用"用户"来称呼！
        3. 如果用户的问题与医学无关，判断用户的目的，并温柔地提示其回到医学话题。
    再次提醒：请严格遵守以上规则，当参考资料不足时，拒绝回答问题，不要自行发挥！"""

    # 欢迎提示词
    greeting_prompt = "你好！[我是Molly医疗精灵]！专注解决你的医疗问题。请问你需要什么帮助？"

    #对话提示词模板
    prompt_template ="""
    ##用户问题：{input}

    ##参考资料：

    ##本地知识库：{rag_results}

    ##对话历史：{chat_history}
    """

class Robot:
    def __init__(self, model_config, retriever=None):
        self.prompts = Prompts()
        
        llm = ChatOpenAI(**model_config)
        
        # template格式化human_message
        template = ChatPromptTemplate.from_messages([("human", self.prompts.prompt_template)])
        
        # 当没有检索器时，设置检索器为人工输入
        if retriever is None:
            retriever = RunnableLambda(lambda input: "")
            
        # 设置输入处理的
        llm_hist = RunnableWithMessageHistory(
            template | llm,
            get_session_history=self.get_history,
            history_messages_key="chat_history"  # chain模板中，传入的chat_history变量在模板中的"key"
        )
        
        self.chain = {'input': RunnablePassthrough(), 'rag_results': retriever, 'chat_history': RunnablePassthrough()} | llm_hist
        
    def check_session_id(self):
        # 判断数据库中的session_id是否存在
        con = sqlite3.connect('chat_history.db')
        cursor = con.cursor()
        
        valid_table_exists_sql = "select count(*) from sqlite_master where type='table' and name='message_store'"
        res = cursor.execute(valid_table_exists_sql)
        
        if res.fetchone()[0] == 0:
            return []
            
        search_session_id_sql = f"select distinct session_id from message_store"
        res = cursor.execute(search_session_id_sql)
        
        # 获取所有session_id
        all_session_id = res.fetchall()
        
        # 关闭数据库连接
        cursor.close()
        con.close()

        return [int(item[0]) for item in all_session_id]

    def get_history(self, session_id):
        if session_id not in self.check_session_id(): 
            history = SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")
            history.add_message(SystemMessage(content=self.prompts.system_prompt))
            history.add_message(AIMessage(content=self.prompts.greeting_prompt))
        return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")
        
    def chat(self, input, session_id):
        config = {'configurable': {'session_id': session_id}}
        response = self.chain.invoke(input, config=config)
        return response.content
        
    def stream(self, inputs, session_id):
        config = {'configurable': {'session_id': session_id}}
        response = self.chain.stream(inputs, config=config)
        return response


if __name__ == "__main__":
    load_dotenv()
    
    robot=Robot(model_config={'model': 'gpt-3.5-turbo'})
    result = robot.chat("如何治疗脑卒中的疾病？", session_id="abc123")
    print("答复：", result)