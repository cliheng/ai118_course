import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

def get_session_messages(session_id:int=None)->list[tuple[str,str]]:
    """
    得到一个对话的所有消息，但是变成元组列表的形式便于解析。第一个元素是角色，第二个元素是消息的内容。
    如果session_id没有指定，则默认使用当前会话。
    :param session_id:
    :return:
    """
    default_session_id = st.session_state.get('session_id', session_id)
    hist_msg = st.session_state['robot'].get_history(session_id=session_id or default_session_id)
    message = []
    # 使用.messages属性获取消息列表
    for msg in hist_msg.messages[1:]:  # 跳过系统消息
        if isinstance(msg, HumanMessage):
            message.append(("Human", msg.content))  # 注意这里是元组
        if isinstance(msg, AIMessage):
            message.append(("AI", msg.content))
    return message

def create_response(question:str,session_id:int=None)->str:
    """
    创建一个对话的响应。
    :param question:
    :param session_id:
    :return:
    """
    session_id = session_id or st.session_state.get('session_id')
    return st.session_state.get('robot').stream(question,session_id)

def start_session():
    """
    创建一个新的对话会话。
    :return:
    """
    max_session_id=max(st.session_state['robot'].check_session_id()+[0])
    st.session_state['session_id'] = max_session_id+1
    st.session_state['robot'].get_history(st.session_state['session_id'])

def get_all_session_id():
    return st.session_state['robot'].check_session_id()

def continue_session(session_id:int=None):
    """
    继续一个对话会话。
    :param session_id:
    :return:
    """
    st.session_state['session_id'] = session_id

def delete_session(session_id:int=None):
    """
    删除一个对话会话。
    :param session_id:
    :return:
    """
    st.session_state['robot'].get_history(session_id=session_id).clear()
    if session_id==st.session_state['session_id']:
        st.session_state['session_id'] = max(st.session_state['robot'].check_session_id()+[0])

