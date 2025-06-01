import gradio as gr
import base64
from zhipuai import ZhipuAI


def chat_proc(message, history=None):
    msg = message['text']
    img_path = message['files'][0]
    with open(img_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')

    client = ZhipuAI(api_key="59251781bb444688a4656508dceccaca.Lo7JWjD9N0XKfIu8") # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4v-plus-0111",  # 填写需要调用的模型名称
        messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": img_base
                }
            },
            {
                "type": "text",
                "text": msg
            }
            ]
        }
        ]
    )
    return response.choices[0].message.content

demo = gr.ChatInterface(
   fn=chat_proc,
   title="图片聊天机器人",
   multimodal=True, 
)

if __name__ == "__main__":
    demo.launch()