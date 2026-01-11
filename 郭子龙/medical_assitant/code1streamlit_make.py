import streamlit as st
import code7_funcs as func
from code6_robot import Robot
from dotenv import load_dotenv
import pyarrow
from code3_chroma import MyChroma

if __name__ == '__main__':


    load_dotenv()
    
    #保存相关公共对象
    if 'started' not in st.session_state:
        #初始化flag对象
        st.session_state = True
        #初始化向量数据库并转换为retriever
        retriever = MyChroma.add_folder('./files/rag','rag_collection','./files/docs').as_retriever()
        #robot对象:管理回话
        st.session_state['robot'] = Robot(model_config={'model':'gpt-3.5-turbo'})
        #session_id:当前回话的id
        st.session_state['session_id'] = 1    #默认回话id为1

    
    st.set_page_config(page_title = "Medical Chatbot",layout = "wide")
    
    st.title("Molly 医疗精灵")

    #查询指定session_id的对话历史
    messages = func.get_session_messages()

    #显示对话历史
    for role,content in messages:
        with st.chat_message(role):
            st.write(content)


    question = st.chat_input("输入问题提问...")

    #根据输入内容项判断是否进行对话
    if question is not None:
        response = func.create_response(question)

        #用户问题添加到聊天窗口chat_message
        st.chat_message('Human').write(question)
        #AI生成回答
        st.chat_message('AI').write_stream(response)



    with st.sidebar:
        st.header(f"当前对话ID:{st.session_state['session_id']}")    #设置侧边栏的标题
        st.button("开始新对话",on_click=func.start_session)  #侧边栏的按钮，点击会触发start_session函数

        #查询所有session_id,添加多个expander
        all_session = func.get_all_session_ids()

        for sid in all_session:
            with st.expander(f"对话ID：{sid}"):
                col1,col2 = st.columns(2)
                #添加重复交互型组件，添加key属性
                col1.button("继续对话",key = f"restart_{sid}",on_click = func.continue_session,args = (sid,))
                col2.button("删除对话",key = f"delete_{sid}",on_click = func.delete_session,args = (sid,))

                #查询指定session_id的对话历史
                message = func.get_session_messages(sid)

                #显示对话历史
                for role,content in messages:
                    with st.chat_message(role):
                        st.write(content)


#uv run streamlit run code1streamlit_make.py