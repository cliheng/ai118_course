import streamlit as st

import funcs
from chroma import MyChroma
from robot import Robot

if __name__ == "__main__":
    if 'started' not in st.session_state:
        st.session_state.started = True
        retriver=MyChroma.add_folder('./files/rag','rag_collection','./files/docs').as_retriever()
        st.session_state['robot'] = Robot(model_config={'model': 'gpt-3.5-turbo'},retriever=retriver)
        st.session_state['session_id'] = 1
    st.set_page_config(page_title="Medical Chatbot", layout="wide")
    st.title("Molly 医疗精灵")
    message=funcs.get_session_messages()
    for role,msg in message:
        with st.chat_message(role):
            st.write(msg)
    question = st.chat_input("输入问题提问......")
    if question is not None:
        res=funcs.create_response(question)
        st.chat_message('Human').write(question)
        st.chat_message('AI').write_stream(res)
    with st.sidebar:
        all_session=funcs.get_all_session_id()
        st.header(f"当前对话ID: {st.session_state['session_id']}")
        st.button("开始新对话",on_click=funcs.start_session())
        for sid in all_session:
            with st.expander(f"对话ID: {sid}"):
                col1, col2 = st.columns(2)
                for role,msg in funcs.get_session_messages(sid):
                    with st.chat_message(role):
                        st.write(msg)
                # 添加重复性的交互主键的时候需要添加额外的一个属性以确保唯一性
                col1.button("继续对话",key=f'restart{sid}',on_click=funcs.continue_session,args=(sid,))
                col2.button("删除对话",key=f'delete{sid}',on_click=funcs.delete_session,args=(sid,))

