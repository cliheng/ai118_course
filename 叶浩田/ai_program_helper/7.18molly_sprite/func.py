
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from sql_history2 import Robot

def get_session_messages(session_id=None) -> list[tuple[str, str]]:
    """
    得到一个对话的所有消息，但是变成元组列表的形式便于解析。第一个元素是角色，第二个元素是消息的内容。
    如果session_id没有指定，则默认使用当前会话。
    """
    default_session_id = st.session_state.get('session_id', 1)
    hist_msg = st.session_state['robot'].get_history(session_id=session_id or default_session_id)

    # 将消息转换为streamlit呈现的消息格式
    messages = []
    for msg in hist_msg.messages[1:]:
        if isinstance(msg, HumanMessage):
            messages.append(("HUMAN", msg.content))
        if isinstance(msg, AIMessage):
            messages.append(("AI", msg.content))

    # print("获取对话历史: ", messages)
    return messages

def create_response(question: str,session_id=None) -> str:

    """
    调用'Robot'类的'stream'方法，得到AI的回复。'stream'方法返回的是流式输出的'Iterator'对象，
    """
    session_id = session_id or st.session_state.get('session_id')
    return st.session_state['robot'].stream(question, session_id=session_id)

    # # 使用机器人处理问题
    # response = st.session_state['robot'].process_input(question)
    
    # # 将响应添加到会话历史中
    # st.session_state['robot'].add_message(AIMessage(content=response))
    
    # return response

def start_session() -> None:
    """
    创建一个新的会话ID，并使用'Robot.get_session()'方法创建或者获取会话对象。
    为了避免会话id重复，我们取所有回答id的最大值加1作为新的会话id。
    """
    max_session_id = max(st.session_state['robot'].check_session_id()+[0])   
    st.session_state['session_id'] = max_session_id + 1  # 重置会话ID
    st.session_state['robot'].get_history(st.session_state['session_id'])

def get_all_session_ids() :
    """
    访问'Robot'类的'session_data'属性，获取所有的会话ID。
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
    if session_id==st.session_state['session_id']:
        all_session_ids = st.session_state['robot'].check_session_id()
        st.session_state['session_id'] = max(all_session_ids, default=1)
    
if __name__ == "__main__":
    st.session_state['robot'] = Robot(model_config={'model': 'gpt-3.5-turbo'})
    # 测试获取对话历史
    # get_session_messages('abc789')  # 使用默认会话ID
    
    resp=create_response("如何治疗脑卒中的疾病？")
    
    for i in resp:
        print(i.content)