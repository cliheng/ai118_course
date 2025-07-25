import streamlit as st
import logging
from uuid import uuid4
from robot import Robot
import func
from dotenv import load_dotenv
from my_chroma import MyChroma

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 加载环境变量
load_dotenv()


def interface():
    # 确保构建的页面的宽度与父级相同
    st.set_page_config(layout="wide")

    if "session_id" not in st.session_state:
        # 在 session_state 中存储 session_id
        st.session_state.session_id = str(uuid4())
    if "robot" not in st.session_state:
        # 使用 MyChroma 类方法将文档文件夹添加到向量数据库
        retriever = MyChroma.add_folder(
            "./files/rag", "rag_collection", "./files/docs"
        ).as_retriever(search_kwargs={"k": 2})
        # 实例化Robot类
        st.session_state.robot = Robot(retriever=retriever)

    # 标题设置
    st.title("Molly 医疗问答智能助手")

    # 侧边栏构建
    with st.sidebar:
        st.header(f"当前对话ID：{st.session_state.session_id}")
        st.button("创建新会话", on_click=func.start_session)

        # 获取所有会话session_id
        session_ids = func.get_all_session_ids()
        # 遍历所有会话session_id，创建折叠块
        for id in session_ids:
            # 创建折叠块
            with st.expander(f"对话ID：{id}"):
                col1, col2 = st.columns(2)
                with col1:
                    # 创建‘继续对话’按钮，点击后调用func.continue_session函数，传入当前会话id
                    st.button(
                        "继续对话",
                        key=f"continue_{id}",
                        on_click=func.continue_session,
                        args=(id,),
                    )
                with col2:
                    # 创建‘删除对话’按钮，点击后调用func.delete_session函数，传入当前会话id
                    st.button(
                        "删除对话",
                        key=f"delete_{id}",
                        on_click=func.delete_session,
                        args=(id,),
                    )
                # 侧边栏记录会话内容展示
                messages = func.get_session_messages(id)
                for msg in messages:
                    with st.chat_message(msg[0]):
                        st.write(msg[1])

    # 构建对话主体
    with st.container():

        # 对话记录展示
        messages = func.get_session_messages(st.session_state.session_id)
        for msg in messages:
            with st.chat_message(msg[0]):
                st.write(msg[1])

    # 用户输入
    question = st.chat_input("请输入问题......")

    # 如果用户输入不为空，则传递给大模型并返回生成内容
    if question is not None:
        # 将用户输入的问题显示到上方
        with st.chat_message("user"):
            st.write(question)
        # 调用大模型的stream方法，传入用户输入和会话ID
        response = func.create_response(question, st.session_state.session_id)
        if response:
            # 将大模型返回的内容展示在页面上
            with st.chat_message("ai"):
                st.write_stream(response)
        else:
            st.error("获取回复失败")


if __name__ == "__main__":
    interface()
