import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from code6_robot import Robot



def get_session_messages(session_id: str = None) -> list[tuple[str, str]]:
    """
    得到一个对话的所有消息，但是变成元组列表的形式便于解析。第一个元素是角色，第二个元素是消息的内容。
    如果session_id没有指定，则默认使用当前会话。
    """
    default_session_id = st.session_state.get('session_id', '1') # 默认值改为字符串'1'
    hist_msg = st.session_state['robot'].get_history(session_id=session_id or default_session_id)

    # 将消息转换为streamlit呈现的消息格式
    messages = []
    for msg in hist_msg.messages[1:]:
        if isinstance(msg, HumanMessage):
            messages.append(("HUMAN", msg.content))
        if isinstance(msg, AIMessage):
            messages.append(("AI", msg.content))

    # print("获取对话历史：", messages)
    return messages

def start_session()->None:
    """
    创建一个新的会话ID,并使用`Robot.get_session()`方法创建或者获取会话对象。
    为了避免会话ID重复,我们取所有回答ID的最大值加1作为新的会话ID。
    """
    all_ids = st.session_state['robot'].check_session_id()
    numeric_ids = [int(sid) for sid in all_ids if sid.isdigit()]
    max_session_id = max(numeric_ids) if numeric_ids else 0
    st.session_state['session_id'] = str(max_session_id + 1) # 新ID也为字符串
    st.session_state['robot'].get_history(st.session_state['session_id'])


def create_response(question:str,session_id: str =None)->str:
    """
    调用`Robot`类的`stream`方法,得到AI的回复。`stream`方法返回的是流式输出的`Iterator`对象,
    """
    session_id=session_id or st.session_state.get('session_id')
    return st.session_state['robot'].stream(question,session_id)

def get_all_session_ids():
    """
    访问`Robot`类的`session_data`属性,得到所有会话的ID和名称的列表。
    """
    return st.session_state['robot' ].check_session_id()

def continue_session(session_id:str)->None:
    """
    将全局变量的session_id设置为指定的会话ID。
    此时,聊天记录显示、和产生模型回复等都会使用新设置的会话ID。
    """
    st.session_state['session_id'] = session_id

def delete_session(session_id:str)->None:
    """
    删除指定的会话ID对应的会话对象。
    同时需要重置session_id,否则在`get_session_messages`中会调用`Robot.get_session()`方法,再创建这个ID的对话。
    """
    st.session_state['robot'].get_history(session_id).clear()
    if session_id == st.session_state['session_id']:
        remaining_ids = st.session_state['robot'].check_session_id()
        if remaining_ids:
            st.session_state['session_id'] = remaining_ids[0]
        else:
            # If no sessions are left, start a new one with ID '1'
            st.session_state['session_id'] = '1'
            st.session_state['robot'].get_history('1')


if __name__ == "__main__":
    st.session_state['robot'] = Robot(model_config={'model': 'gpt-3.5-turbo'})
    # 测试获取对话历史
    resp = create_response("脑卒中发生后，需注意哪些方面？")

    for i in resp:
        print(i.content)