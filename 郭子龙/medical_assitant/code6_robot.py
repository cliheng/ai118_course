from langchain_core.runnables import RunnableWithMessageHistory,RunnableLambda,RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate
import os
import sqlite3
from dotenv import load_dotenv


class Prompts:
    # 系统提示词
    system_promt = """你是一个名叫Molly的医学专家，
            对于用户提问的医学相关问题，你需要按照给出的参考文献资料对问题进行回答。
            你的回答需要按照以下步骤：
                1. 分析用户问题、对话历史以及参考文献，判断参考资料的哪些内容可以解答用户的问题，并将这一过程进行说明。
                2. 如果参考文献可以解答用户的问题，则根据文献内容对问题进行解答。
                3. 如果参考文献不能解答用户问题，告诉用户信息不足，无法回答，建议用户寻求专业人士帮助，不要自行发挥。
            你的回答需要注意以下几点： 
                1. 保证你的回答是清晰的、明确的。如果你参考了参考资料，应该指出参考资料的标题等。
                2. 结合用户的对话历史，分析用户的问题意图。但不要复述问题。
                2. 回复用户时，使用对话的口吻，有礼貌地称呼用户为”您“，不要使用“用户”来称呼！
                3. 如果用户的问题与医学无关，判断用户的目的，并温柔地提示其回到医学话题。
            再次提醒：请严格遵守以上规则，当参考资料不足时，拒绝回答问题，不要自行发挥！"""


    # 欢迎提示词   
    greeting_prompt = "你好！我是Molly医疗精灵，专注解决你的医疗问题。请问你需要什么帮助？"
    # 对话提示词模版
    prompt_template = """##用户问题：{input}
        
    ##参考资料：

    ##本地知识库：{rag_results} 

    ##对话历史：{chat_history}"""

class Robot:

    def __init__(self,model_config,retriever = None):
        self.prompts = Prompts()
        llm = ChatOpenAI(**model_config)
    
        #template填充的human_message
        template = ChatPromptTemplate.from_messages(["human",self.prompts.prompt_template])
    
        #通过表达式方式，实现更多数据传入
        if retriever is None:
            retriever = RunnableLambda(lambda input:"")
    
        #管理聊天历史
        llm_hist = RunnableWithMessageHistory(
            template | llm,
            get_session_history = self.get_history,
            history_messages_key = "chat_history"
        )
        self.chain = {
            'input': RunnablePassthrough(),
            'rag_results': retriever,
            'chat_history': RunnablePassthrough()
        } | llm_hist


    def check_session_id(self):

    #连接数据库
        con = sqlite3.connect("chat_history.db")
        #创建游标
        cursor = con.cursor()
        #验证表是否存在
        #*代表表中所有列
        #count(*)代表查询结果的行数
        #sqlite_master 是一个系统表，存储了数据库的所有表和索引
        search_session_id_sql = "select count(*) from sqlite_master where type = 'table' and name = 'message_store'"
        res = cursor.execute(search_session_id_sql)

        
        if res.fetchone()[0] == 0:    # 第一次连接数据库，表还未创建

            return []
    
            
        #查询指令并运行  select 列名 from 表名 [where 条件]
        #search_session_id_sql = "select idmsession_id,message from message_store"
        search_session_id_sql = f"select distinct session_id from message_store"
        res = cursor.execute(search_session_id_sql)

        #获取查询结果
        # session_id = res.fetchone()
        all_session_id = res.fetchall()

        #关闭游标和连接
        cursor.close()
        con.close()

        return [int(item[0]) for item in all_session_id]


    def get_history(self,session_id):
        if session_id not in self.check_session_id():
            history = SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")
            #如果session_id不存在，则添加系统提示和欢迎提示
            history.add_message(SystemMessage(content = self.prompts.system_promt))
            history.add_message(AIMessage(content = self.prompts.greeting_prompt))
        return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

    def chat(self,input,session_id):
        config = {'configurable':{'session_id':session_id}}
        response = self.chain.invoke(input,config = config)
        return response.content
        
    def stream(self,input,session_id):
        config = {'configurable':{'session_id':session_id}}
        response =self.chain.stream(input,config = config)
        return response




if __name__ == '__main__':
    load_dotenv()
    robot = Robot(model_config = {'model':'gpt-3.5-turbo'})
    result = robot.chat('你能帮我找找附近的美食吗',session_id = 'abc123')
    print("答复：",result)
  
