import streamlit as st
from sql_chat_bot_6 import Robot
import code_7_func as func
from dotenv import load_dotenv
load_dotenv()
from code_3_chroma import MyChroma
# 保存相关公共对象
if'started' not in st.session_state:
    # 初始化flag对象
    st.session_state.started = True
    #初始化向量数据库并转化为retriever
    retriever = MyChroma.add_folder('./files/rag', 'rag_collection', 'files/docs').as_retriever()
    # robot对象：管理会话
    st.session_state['robot'] = Robot(model_config={'model': 'gpt-3.5-turbo'})
    # session_id：当前会话的ID
    st.session_state['session_id'] = 1  # 默认会话ID为1
st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("Molly 医疗精灵")
# 查询指定session_id的对话历史
messages = func.get_session_messages()
# 显示对话历史
for role, content in messages:
    with st.chat_message(role):
        st.write(content)
questions= st.chat_input("请输入您的问题：")
if questions is not None:
    response=func.create_response(questions)
    #用户问题添加到聊天窗口chat_message
    st.chat_message("HUMAN").write(questions)
    st.chat_message("AI").write_stream(response)
with st.sidebar:
    st.header(f"当前对话ID:{st.session_state['session_id']}")
    st.button("开始新的对话",on_click= func.start_session)
    #查询
    all_session = func.get_all_session_ids()
    for sid in all_session:
        with st.expander(f"对话ID:{sid}"):
            col1,col2=st.columns(2)
            col1.button("继续对话",key=f"restart_{sid}",on_click=func.continue_session,args=(sid,) )
            col2.button("删除对话",key=f"delete_{sid}",on_click=func.delete_session,args=(sid,) )
    #查询指定ID的历史对话
    messages = func.get_session_messages(sid)
    #显示历史对话
    for role, content in messages:
        with st.chat_message(role):
            st.write(content)
