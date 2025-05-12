import gradio as gr
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-46b5f9f6a77b435d8d21d23ff920db0d",
    base_url="https://api.deepseek.com"
)

def chat_ai(user_prompt: str, chat_history: list = None) -> str:
    """
    基于用户提示和聊天历史生成AI回复
    
    参数:
        user_prompt (str): 用户的输入提示信息
        chat_history (list, optional): 聊天历史记录列表
    
    返回:
        str: AI生成的回复文本
    """
    try:
        # 准备消息列表
        messages = [
            {"role": "system", "content": "你是一个有帮助的AI助手。"}
        ]
        
        # 添加历史记录
        if chat_history:
            messages.extend(chat_history)
        
        # 添加用户当前输入
        messages.append({"role": "user", "content": user_prompt})
        
        # 调用API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"发生错误：{str(e)}"

def gradio_chat(message, history):
    """
    Gradio聊天界面处理函数
    """
    # 将Gradio的历史记录格式转换为API需要的格式
    api_history = []
    for human, ai in history:
        api_history.append({"role": "user", "content": human})
        api_history.append({"role": "assistant", "content": ai})
    
    # 获取AI回复
    response = chat_ai(message, api_history)
    return response

# 创建Gradio界面
demo = gr.ChatInterface(
    fn=gradio_chat,
    title="AI聊天助手",
    description="这是一个基于DeepSeek API的AI聊天助手，支持多轮对话。",
    theme="soft",
    examples=[
        "你好，请介绍一下你自己",
        "今天天气怎么样？",
        "你能帮我写一首诗吗？"
    ]
)

# 启动界面
if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
