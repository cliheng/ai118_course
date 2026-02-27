import streamlit as st
import logging
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage


def get_session_messages(session_id: str):
    """
    得到一个对话的所有消息，但是变成元组列表的形式便于解析。第一个元素是角色，第二个元素是消息的内容。
    如果session_id没有指定，则默认使用当前会话。
    """
    if session_id:
        # 指定对应的 session_id
        session = st.session_state.robot.get_history(session_id)
        # 获取返回的所有对话内容并转化为元组格式
        messages = []
        for msg in session.messages:
            # isinstance 判断当前信息是否是 HumanMessage 类型
            if isinstance(msg, HumanMessage):
                messages.append(("human", msg.content))
            # isinstance 判断当前信息是否是 AIMessage 类型
            elif isinstance(msg, AIMessage):
                messages.append(("ai", msg.content))

        logging.info(f"获取会话ID {session_id} 的所有消息")
        # 将处理好的指定session_id的历史信息列表返回
        return messages
    else:
        return []


def get_all_session_ids() -> list[str]:
    """
    访问`Robot`类的`check_session_ids`属性，得到所有会话的ID和名称的列表。
    """
    logging.info("获取所有会话ID")
    # 调用Robot中的check_session_ids方法获取所有会话session_id
    return st.session_state.robot.check_session_ids()


def create_response(question: str, session_id: str):
    """
    调用`Robot`类的`stream`方法，得到AI的回复。`stream`方法返回的是流式输出的`Iterator`对象，需要使用streamlit.stream_write()方法输出。
    """
    if session_id:
        response = st.session_state.robot.stream(question, session_id)
        if response:
            logging.info(f"获取会话ID {session_id} 的回复")
            return response
        else:
            logging.error("获取回复失败")
            return False
    else:
        logging.error("没有指定会话ID，无法获取回复")
        return False


def start_session() -> None:
    """
    创建一个新的会话ID，并使用`Robot.get_session()`方法创建或者获取会话对象。
    """
    # 重新生成session_id，触发streamlit的实时更新
    st.session_state.session_id = str(uuid4())
    logging.info(f"创建新会话ID {st.session_state.session_id}")


def continue_session(session_id: str) -> None:
    """
    将全局变量的session_id设置为指定的会话ID。
    此时，聊天记录显示、和产生模型回复等都会使用新设置的会话ID。
    """
    if session_id:
        st.session_state.session_id = session_id
        logging.info(f"设置当前会话ID为 {session_id}")


def delete_session(session_id: str) -> None:
    """
    删除指定的会话ID对应的会话对象。
    同时需要重置session_id，否则在`get_session_messages`中会调用`Robot.get_session()`方法，再创建这个ID的对话。
    """
    if session_id:
        # 将对应session_id的信息调用get_history中的clear方法
        st.session_state.robot.get_history(session_id).clear()
        st.info(f"会话ID {session_id} 已删除")
        # 如果session_id是当前会话id，则新创建一个会话
        if session_id == st.session_state.session_id:
            st.session_state.session_id = str(uuid4())
