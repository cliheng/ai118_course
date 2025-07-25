from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough, RunnableLambda
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
import os
import sqlite3
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

load_dotenv()

def format_docs(docs):
    """Formats a list of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

class Prompts:
    #系统提示词
    system_prompt="""
        你是一个名叫Molly的医疗精灵，
        对于患者的问题，你需要按照给出的参考文献给出回答。
        你的回答需要按照一下步骤：
            1. 分析用户问题、对话历史以及参考文献,判断参考资料的哪些内容可以解答用户的问题,并将这一过程进行说明。
            2. 如果参考文献可以解答用户的问题,则根据文献内容对问题进行解答。
            3. 如果参考文献不能解答用户问题,告诉用户信息不足,无法回答,建议用户寻求专业人士帮助,不要自行发挥。
        你的回答要注意一下几点：
            1. 保证你的回答是清晰的、明确的。如果你参考了参考资料,应该指出参考资料的标题等。
            2. 结合用户的对话历史,分析用户的问题意图。但不要复述问题。
            3. 回复用户时,使用对话的口吻,有礼貌地称呼用户为”您“,不要使用“用户”来称呼!
            4. 如果用户的问题与医学无关,判断用户的目的,并温柔地提示其回到医学话题。
        再次强调,请严格按照以上步骤和要求回答用户的问题，当参考文献不能解答用户问题时，不要自行发挥。
        """
    #欢迎提示词
    greeting_prompt="你好！我是Molly医疗精灵，专注为患者提供医疗咨询服务。"

    prompt_template="""##用户问题：{input}
    ##参考文献：

    ##本地知识库：{rag_result}

    ##对话历史：{chat_history}
    """

class Robot:
    def __init__(self,model_config,retriever=None):
        self.prompts=Prompts()
        llm = ChatOpenAI(**model_config)
        template = ChatPromptTemplate.from_messages([("human", self.prompts.prompt_template)])

        prompt_and_llm = template | llm
        
        # The chain now formats the documents from the retriever into a string.
        chain_with_rag = RunnablePassthrough.assign(
            rag_result=(itemgetter("input") | retriever | RunnableLambda(format_docs))
        ) | prompt_and_llm

        # Use RunnableWithMessageHistory to wrap the chain for automatic history management
        self.chain = RunnableWithMessageHistory(
            chain_with_rag,
            self.get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def check_session_id(self):
        con = sqlite3.connect('chat_history.db')
        cursor = con.cursor()

        valid_table_exists_sql=f"select count(*) from sqlite_master where type='table' and name='message_store'"
        res=cursor.execute(valid_table_exists_sql)

        if res.fetchone()[0] == 0:
            return []
        search_session_id_sql=f"select distinct session_id from message_store"
        res=cursor.execute(search_session_id_sql)
        
        all_session_id = res.fetchall()

        cursor.close()
        con.close()
        return [item[0] for item in all_session_id if item[0] is not None]

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        session_id_str = str(session_id)
        history = SQLChatMessageHistory(session_id_str, "sqlite:///chat_history.db")
        if not history.messages:
            history.add_message(SystemMessage(content=self.prompts.system_prompt))
            history.add_message(AIMessage(content=self.prompts.greeting_prompt))
        return history

    def chat(self,input,session_id):
        config = {"configurable": {'session_id': str(session_id)}}
        response=self.chain.invoke({"input": input},config=config)
        return response.content

    def stream(self,input,session_id):
        config = {"configurable": {'session_id': str(session_id)}}
        response=self.chain.stream({"input": input},config=config)
        return response

if __name__ == "__main__":
    # This is a dummy retriever for testing purposes.
    # In the actual application, the retriever is created in main.py
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    class DummyRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
            return [Document(page_content=f"This is a dummy document about {query}.")]
    
    robot=Robot(model_config={"model":"gpt-3.5-turbo"}, retriever=DummyRetriever())
    result=robot.chat("我叫什么名字？","abc1234")
    print("答复：",result)