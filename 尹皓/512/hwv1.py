import gradio as gr
from openai import OpenAI
client = OpenAI(api_key="sk-e1c976043cbf4bf5b66e4fd79efe3f1c", base_url="https://api.deepseek.com")

while True:
    user_input = input("请输入您的问题/Please enter your question: ")
  # 替换为您的模型名称
    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个聊天机器人,致力于聊天"},
        {"role": "user", "content": user_input},
    ],
# 初
)
    print(response.choices[0].message.content)
    if user_input == "拜拜":
        break