import gradio as gr
import base64
from zhipuai import ZhipuAI
import os

def chat_proc(message, files, history):
    try:
        client = ZhipuAI(api_key="04d2bda59ba34e2388ad33c3541a9e65.4meytIG8TRAS12pC")
        
        # 根据是否有图片构建不同的请求内容
        if files:
            # 处理图片文件
            if isinstance(files, str):
                img_path = files
            else:
                img_path = files.name if hasattr(files, 'name') else files
                
            if not os.path.exists(img_path):
                return "无法访问图片文件"
                
            with open(img_path, 'rb') as img_file:
                img_base = base64.b64encode(img_file.read()).decode('utf-8')
                
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base}"
                    }
                },
                {
                    "type": "text",
                    "text": message
                }
            ]
        else:
            # 仅文本对话
            content = [
                {
                    "type": "text",
                    "text": message
                }
            ]
            
        response = client.chat.completions.create(
            model="glm-4v-plus-0111",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"处理过程中出现错误：{str(e)}"

# 使用 Blocks 接口
with gr.Blocks() as demo:
    gr.Markdown("## 图文问答机器人")
    gr.Markdown("向机器人提问，它将根据您的问题提供详细的解答。")
    
    with gr.Row():
        # 修改文件组件配置
        file_output = gr.File(
            label="上传图片",
            file_types=["image"],
            type="filepath"  # 指定返回文件路径
        )
        text_input = gr.Textbox(label="输入问题")
    
    text_output = gr.Textbox(label="回答")
    
    submit_btn = gr.Button("提交")
    submit_btn.click(
        fn=chat_proc,
        inputs=[text_input, file_output],
        outputs=[text_output]
    )
    
    gr.Examples(
        examples=[
            ["这张图片是什么内容？", None],
            ["图中有几个人？", None]
        ],
        inputs=[text_input, file_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)