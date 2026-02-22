import streamlit as st
from code3_chroma import MyChroma
from code6_robot import Robot
import code7_func as func

if __name__ == "__main__":
    # 保存相关公共对象
    if 'started' not in st.session_state:
        # 初始化flag对象
        st.session_state.started = True 
        
        retriever=MyChroma.add_folder('./files/rag','rag_collection','./files/rag').as_retriever()
        # robot对象：管理会话
        st.session_state['robot'] = Robot(model_config={'model': 'gpt-3.5-turbo'},retriever=retriever)
        # session_id：当前会话的ID
        st.session_state['session_id'] = '1'  # 修复：默认会话ID为字符串'1'

    st.set_page_config(
        page_title="Medical Chatbot",
        layout="wide"
    )
    
    st.title("MOLLY医疗精灵")
    
    # 查询指定session_id的对话历史
    messages = func.get_session_messages()

    # 显示对话历史
    for role,content in messages:
        with st.chat_message(role):
            st.write(content)
            

    question = st.chat_input("请输入你的问题")

    if question is not None:

        response=func.create_response(question)

        st.chat_message("Human").write(question)

        st.chat_message("AI").write_stream(response)

    with st.sidebar:
        st.header(f"当前对话ID：{st.session_state['session_id']}")
        st.button(f"开始新对话",on_click=func.start_session)
        #查询所有会话ID
        all_session_ids=func.get_all_session_ids()

        for sid in all_session_ids:
            with st.expander(f"对话ID：{sid}"):
                col1, col2 = st.columns(2)
                col1.button(f"继续对话",key=f"restart_{sid}",on_click=func.continue_session,args=(sid,))
                col2.button(f"删除对话",key=f"delete_{sid}",on_click=func.delete_session,args=(sid,))

        with st.expander(f"对话ID：{1}"):
            col1, col2 = st.columns(2)
            col1.button("继续对话")
            col2.button("删除对话")
