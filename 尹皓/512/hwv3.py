import openai
import gradio as gr

def chat_ai(user_prompt, chat_history):
    # API配置参数
    api_key = "sk-e1c976043cbf4bf5b66e4fd79efe3f1c"
    base_url = "https://api.deepseek.com"
    model = "deepseek-reasoner"
    
    messages = [{"role": "system", "content": "你是一个专业的回答问题的AI,致力于详细的回答"}]
    messages.extend({"role": role, "content": content} for role, content in chat_history)
    messages.append({"role": "user", "content": user_prompt})
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content

def handle_chat(message, history):
    converted_history = []
    for human, assistant in history:
        converted_history.extend([
            {"role": "user", "content": human},
            {"role": "assistant", "content": assistant}
        ])
    return chat_ai(message, converted_history)

demo = gr.ChatInterface(
    fn=handle_chat,
    title="DeepSeek智能助手",
    
)

if __name__ == "__main__":
    demo.launch()