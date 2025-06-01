import gradio as gr
import random
from zhipuai import ZhipuAI

def random_response(message, history=None):

    client = ZhipuAI(api_key="59251781bb444688a4656508dceccaca.Lo7JWjD9N0XKfIu8")  # 请填写您自己的APIKey
    response = client.chat.completions.create(
    model="glm-4-air-250414",  # 请填写您要调用的模型名称
    messages=[
        {"role": "user", "content": "message"},
    ],
)
    return response.choices[0].message.content
demo = gr.ChatInterface(
    fn=random_response,
    type='messages',
    title="随机回复机器人",
    description="这是一个简单的随机回复机器人，每次发送消息都会得到一个随机回复。",
    examples=[
        ["你好"],
        ["今天天气怎么样？"],
        ["你喜欢什么颜色？"]
    ],
    cache_examples=True,
)


if __name__ == "__main__":
    demo.launch()