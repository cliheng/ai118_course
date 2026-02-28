import streamlit as st

from sql_history2 import Robot
import func as func

if __name__ == "__main__":

    # 保存相关公共对象
    if 'started' not in st.session_state:
        # 初始化flag对象
        st.session_state.started = True
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

    # with st.chat_message("AI"):
    #     st.write("你好我是Molly医疗精灵，专注于解决你的问题！")
    # with st.chat_message("HUMAN"):
    #     st.write("如何治疗脑卒中的疾病？")