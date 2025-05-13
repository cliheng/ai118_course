import gradio as gr
import openai

def deepseek_v3_model(prompt: str, history: list) -> str:
    # 更新为V3接口参数
    api_key = "sk-c83abe1c508044b4b50fdd1743221a5b"
    base_url = "https://api.deepseek.com"  # 接口版本升级
    model = "deepseek-chat"  # 模型名称更新
    
    # 保留消息构造逻辑
    messages = [{"role": "system", "content": "你是一个专业的回答问题的AI，致力于详细的回答"}]
    for human, assistant in history:
        messages.extend([
            {"role": "user", "content": human},
            {"role": "assistant", "content": assistant}
        ])
    messages.append({"role": "user", "content": prompt})
    
    # 创建API客户端
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    # 新增V3模型参数
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        top_p=0.9,  # 新增多样性控制
        max_tokens=2000  # 增加响应长度
    )
    return response.choices[0].message.content

# ... 保留原有respond函数和Gradio界面配置 ...
def respond(message: str, history: list):
    """Gradio交互处理函数"""
    return deepseek_v3_model(message, history)

demo = gr.ChatInterface(
    respond,
    title="DeepSeek V3 专业版",
    examples=["请解释量子计算原理", "如何实现分布式训练？"],
    chatbot=gr.Chatbot(height=500)
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
