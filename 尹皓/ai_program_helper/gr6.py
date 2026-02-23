import gradio as gr
import base64
from zhipuai import ZhipuAI

def chat_proc(message, dialog_type, history=None):
    client = ZhipuAI(api_key="11f04e62cc3948ff9c40a3a83e8b184a.dINvlam3XSkoMKK3")
    if dialog_type == "图片对话":
        img_path = message['files'][0]
        with open(img_path, 'rb') as img_file:
            img_base = base64.b64encode(img_file.read()).decode('utf-8')
        response = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_base}},
                    {"type": "text", "text": '请描述图片内容'}
                ]
            }]
        )
        return response.choices[0].message.content
    else:  # 文本对话
        text = message
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }]
        )
        return response.choices[0].message.content

with gr.Blocks() as demo:
    gr.Markdown("# Chatbot")
    dialog_type = gr.Radio(["图片对话", "文本对话"], value="图片对话", label="对话类型")
    with gr.Row():
        img_input = gr.File(label="上传图片")
        text_input = gr.Textbox(label="输入文本")
    output = gr.Textbox(label="回复")

    def submit_fn(img, text, dialog_type):
        if dialog_type == "图片对话" and img is not None:
            message = {'files': [img.name]}
            return chat_proc(message, dialog_type)
        elif dialog_type == "文本对话" and text:
            return chat_proc(text, dialog_type)
        else:
            return "请根据对话类型上传图片或输入文本"

    submit_btn = gr.Button("发送")
    submit_btn.click(
        submit_fn,
        inputs=[img_input, text_input, dialog_type],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()