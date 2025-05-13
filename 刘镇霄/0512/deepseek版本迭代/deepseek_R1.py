import gradio as gr
import openai

def deepseek_v2_model(prompt: str, history: list) -> str:
    # API配置参数（生产环境建议使用环境变量）
    api_key = "sk-c83abe1c508044b4b50fdd1743221a5b"
    base_url = "https://api.deepseek.com"
    model = "deepseek-reasoner"
    
    # 构建消息列表（保留原始模板的系统消息）
    messages = [{"role": "system", "content": "你是一个专业的回答问题的AI，致力于详细的回答"}]
    
    # 转换Gradio历史记录格式
    for human, assistant in history:
        messages.extend([
            {"role": "user", "content": human},
            {"role": "assistant", "content": assistant}
        ])
    messages.append({"role": "user", "content": prompt})
    
    # 创建API客户端
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    # 获取模型响应（保持流式关闭）
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        stream=False
    )
    return response.choices[0].message.content

def respond(message: str, history: list):
    """Gradio交互处理函数（保持参数结构不变）"""
    return deepseek_v2_model(message, history)

# 创建增强版Gradio界面
demo = gr.ChatInterface(
    fn=respond,
    title="DeepSeek R1 专业版",
    examples=["请解释量子隧穿效应", "如何实现RAG系统？", "写一个PyTorch训练模板"],
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="输入您的问题...", container=False, scale=7)
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
