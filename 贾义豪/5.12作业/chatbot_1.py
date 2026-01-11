from openai import OpenAI
client = OpenAI(api_key="sk-46b5f9f6a77b435d8d21d23ff920db0d", base_url="https://api.deepseek.com")
while True:
    user_input = input("你说：")
    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个聊天机器人"},
        {"role": "user", "content": user_input}]
)
    print(response.choices[0].message.content)
    if user_input == "退出":
        break

