import streamlit as st
from chroma import MyChroma
from dotenv import load_dotenv

from sql_history2 import Robot
import func as func

if __name__ == "__main__":
    load_dotenv()
    
    # 保存相关公共对象
    if 'started' not in st.session_state:
        # 初始化flag对象
        st.session_state.started = True
        #初始化向量数据库，转化为retriever
        retriever = MyChroma.add_folder('./files/rag','rag_collection','./files/docs').as_retriever()

        # robot对象：管理会话
        st.session_state['robot'] = Robot(model_config={'model': 'gpt-3.5-turbo'})
        # session_id：当前会话的ID
        st.session_state['session_id'] = 1  # 默认会话ID为1

    # 设置 Streamlit 页面的配置，包括页面标题和布局方式
    st.set_page_config(page_title="Medical Chatbot", layout="wide")
    
    # 在页面上显示主标题
    st.title("Molly 医疗精灵")
    # 查询指定session_id的对话历史
    messages = func.get_session_messages()

    # 显示对话历史
    for role, content in messages:
        with st.chat_message(role):
            st.write(content)
    
    # 以 AI 角色显示聊天消息
    with st.chat_message("AI"):
        st.write("你好我是Molly医疗精灵，专注于解决你的问题！")
    # 以人类角色显示聊天消息
    with st.chat_message("HUMAN"):
        st.write("如何治疗脑卒中的疾病？")
    
    # 创建一个聊天输入框，提示用户输入问题
    question = st.chat_input("输入问题提问....")
    
    if question is not None:
        
        response = func.create_response(question)
    
        st.chat_message("Human").write(question)
        st.chat_message("AI").write_stream(response)
 
    # 创建侧边栏
    with st.sidebar:
        # 在侧边栏设置标题，显示当前对话 ID
        st.header(f"当前对话ID：{st.session_state['session_id']}")
        # 在侧边栏创建一个按钮，用于开始新对话
        st.button("开始新对话", on_click=func.start_session)
        
        all_session = func.get_all_session_ids()
        
        for sid in all_session:
            with st.expander(f"对话ID：{sid}"):

                # 将该区域划分为两列
                col1, col2 = st.columns(2)
                # 在第一列创建一个按钮，用于继续对话
                col1.button("继续对话", key=f"continue_{sid}", on_click=func.continue_session, args=(sid,))
                # 在第二列创建一个按钮，用于删除对话
                col2.button("删除对话", key=f"delete_{sid}", on_click=func.delete_session, args=(sid,))
                
                messages = func.get_session_messages(sid)
                
                for role, content in messages:
                    with st.chat_message(role):
                        st.write(content)
