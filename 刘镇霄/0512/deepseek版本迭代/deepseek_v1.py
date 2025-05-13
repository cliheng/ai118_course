import openai
import gradio as gr

# 定义API调用函数
def chat_ai(user_prompt, chat_history):
    # API配置参数（建议后续改为从配置读取
    api_key = "sk-c83abe1c508044b4b50fdd1743221a5b"
    # 基础配置参数
    base_url = "https://api.deepseek.com/v1"
    # 模型名称
    model = "deepseek-reasoner"
    # 构建消息列表（系统消息 + 历史对话 + 新提问）
    messages = [{"role": "system", "content": "你是一个专业的回答问题的AI，致力于详细的回答"}]
    # 将历史对话转换为API需要的格式，目的是将历史对话记录转换为模型能理解的格式，并保留上下文信息
    messages.extend({"role": role, "content": content} for role, content in chat_history)
    # 添加新的用户提问
    messages.append({"role": "user", "content": user_prompt})
    # 创建API客户端
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    # 获取模型响应
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content

# 新增的Gradio界面处理函数
def handle_chat(message, history):
    # 将Gradio格式的历史记录转换为API需要的格式
    converted_history = []
    # 遍历历史记录，将其转换为API需要的格式
    for human, assistant in history:
        converted_history.extend([
            ("user", human),
            ("assistant", assistant)
        ])
    # 获取AI回复
    response = chat_ai(message, converted_history)
    return response

# 创建Gradio聊天界面
demo = gr.ChatInterface(
    fn=handle_chat,
    title="DeepSeek智能助手",
    examples=["请解释量子计算", "如何制作蛋糕?", "Python的GIL是什么？"]
)
# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)