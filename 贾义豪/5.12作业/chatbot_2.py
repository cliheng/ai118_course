from openai import OpenAI
client = OpenAI(api_key="sk-46b5f9f6a77b435d8d21d23ff920db0d", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个聊天机器人"},
        ]
)


def generate_response(prompt, history=None):
    """
    生成基于 DeepSeekAPI 的聊天回复
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
 

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
    )
 
    reply = response.choices[0].message.content
    if not history:
        history = []
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": reply})
    return reply, history

def main():
    """
    控制台交互主函数
    """
    print("欢迎使用聊天机器人！输入 'exit' 退出对话。")
    conversation_history = []  # 存储会话历史
 
    while True:
        user_input = input("你：")
        if user_input.lower() == "exit":
            print("机器人：再见！")
            break
 
        reply, conversation_history = generate_response(user_input, conversation_history)
        print(f"机器人：{reply}")

if __name__ == "__main__":
    main()
