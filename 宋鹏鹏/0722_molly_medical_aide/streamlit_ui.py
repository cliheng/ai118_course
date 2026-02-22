import streamlit as st
from robot import Robot
import funcs as func
from dotenv import load_dotenv
import os

from chroma_emb import MyChroma

load_dotenv()

if __name__ == "__main__":

    # 保存相关公共对象
    if 'started' not in st.session_state:   
        # 初始化flag对象
        st.session_state.started = True
        
        # 初始化Chroma数据库,并转换为retriever对象
        retriever = MyChroma.add_folder('./files/rag', 'rag_collection', './files/docs').as_retriever()
        # robot对象：管理会话
        st.session_state['robot'] = Robot(model_config={
        'model': 'gpt-3.5-turbo',
        'api_key': os.getenv('OPENAI_API_KEY'),
        'base_url': os.getenv('OPENAI_API_BASE')
        })
        # session_id：当前会话的ID
        st.session_state['session_id'] = 1  # 默认会话ID
    

    st.set_page_config(page_title="Medical Chatbot", layout="wide")
    st.title("Molly 医疗精灵")
    

    # 查询指定session_id的对话历史
    messages = func.get_session_messages()

    # 展示对话历史
    for role, content in messages:
        with st.chat_message(role):
            st.write(content)

    # with st.chat_message("AI"):
    #     st.write("你好我是Molly医疗精灵，专注于解决你的问题！")
    # with st.chat_message("HUMAN"):
    #     st.write("如何治疗脑卒中的疾病？")

    question = st.chat_input("输入问题提问....")
    #根据输入项判断是否进行对话
    if question is not None:
        response = func.create_response(question, session_id=st.session_state['session_id'])
        st.chat_message("human").write(question)
        st.chat_message("AI").write_stream(response)

    with st.sidebar:
        st.header(f"当前对话ID：{st.session_state['session_id']}")  # 设置侧边栏的标题
        st.button("开始新对话", on_click=func.start_session)  # 侧边栏的按钮，点击后会触发start_session函数

        all_session_ids = func.get_all_session_ids()

        for sid in all_session_ids:
            with st.expander(f"会话ID: {sid}"):
                col1, col2 = st.columns(2)
                col1.button("继续对话",key=f"restart_{sid}",on_click=func.continue_session, args=(sid,))
                col2.button("删除对话",key=f"delete_{sid}", on_click=func.delete_session, args=(sid,))

                # 查询指定session_id的对话历史
                messages = func.get_session_messages(session_id=sid)
                for role, content in messages:
                    with st.chat_message(role):
                        st.write(content)