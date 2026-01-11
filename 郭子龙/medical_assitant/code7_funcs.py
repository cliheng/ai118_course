import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
# Python 模块名不能以数字开头，推测正确模块名可能是在前面加下划线或其他合法字符，这里假设为 _6_robot
from code6_robot import Robot
 
def get_session_messages(session_id: int=None) -> list[tuple[str, str]]:
    """
    得到一个对话的所有消息，但是变成元组列表的形式便于解析。第一个元素是角色，第二个元素是消息的内容。
    如果session_id没有指定，则默认使用当前会话。
    """
    default_session_id = st.session_state.get('session_id','abc')
    # 移除错误的connection参数
    hist_msg = st.session_state['robot'].get_history(session_id=session_id or default_session_id)

    #将消息转换成为streamlit呈现的消息格式
    messages = []
    for msg in hist_msg.messages[1:]:
        if isinstance(msg,HumanMessage):
            messages.append(("HUMAN",msg.content))
        if isinstance(msg,AIMessage):
            messages.append(("AI",msg.content))
    return messages


def create_response(question: str, session_id = None) -> str:
    """
    调用`Robot`类的`stream`方法，得到AI的回复。`stream`方法返回的是流式输出的`Iterator`对象，需要使用streamlit.stream_write()方法输出。
    """
    session_id = session_id or st.session_state.get('session_id')
    return st.session_state.get('robot').stream(question,session_id)

def start_session() -> None:
    """
    创建一个新的会话ID，并使用`Robot.get_session()`方法创建或者获取会话对象。
    为了避免会话ID重复，我们取所有回答ID的最大值加1作为新的会话ID。
    """
    #创建一个新的回话id(原有session_id + 1)
   
    max_session_id =max(st.session_state['robot'].check_session_id() + [0])
    st.session_state['session_id'] = max_session_id + 1

    st.session_state['robot'].get_history(st.session_state['session_id'])


def get_all_session_ids() -> list[tuple[int, str]]:
    """
    访问`Robot`类的`session_data`属性，得到所有会话的ID和名称的列表。
    """
    return st.session_state['robot'].check_session_id()


def continue_session(session_id: int) -> None:
    """
    将全局变量的session_id设置为指定的会话ID。
    此时，聊天记录显示、和产生模型回复等都会使用新设置的会话ID。
    """
    st.session_state['session_id'] = session_id
    
def delete_session(session_id: int) -> None:
    """
    删除指定的会话ID对应的会话对象。
    同时需要重置session_id，否则在`get_session_messages`中会调用`Robot.get_session()`方法，再创建这个ID的对话。
    为了简单，将会话ID重置为所有会话ID的最大值。
    """
    st.session_state['robot'].get_history(session_id).clear()

    if session_id == st.session_state['session_id']:
        st.session_state['session_id'] = max(st.session_state['robot'].check_session_id())




if __name__ == "__main__":
   st.session_state['robot'] = Robot(model_config = {'model':'gpt-3.5-turbo'})
   #测试获取对话历史
   get_session_messages('abc123')  #使用默认回话id
    