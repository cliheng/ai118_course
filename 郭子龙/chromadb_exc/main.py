import json
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from modelscope import AutoTokenizer, AutoModel
import torch
import chromadb
from zhipuai import ZhipuAI
import gradio as gr
import openai
import numpy as np
import requests
import base64

# 这个函数将被删除，因为已经被新的 llm_chat_with_file 替代
def local_embdding(sentences):
    #tokenizer输入文本转换模型输入需要变量类型
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    model.eval()

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    #生成embedding
    with torch.no_grad():
        model_output = model(**encoded_input)
        #从结果中抽取模型生成embedding
        sentence_embeddings = model_output[0][:, 0]

    sentence_embeddings = sentence_embeddings.numpy().tolist()
    #print('Sentence embddings:',len(sentence_embeddings))
    return sentence_embeddings
def load_qa_data(data):
    #处理后数据存储list
    keywords,contents = [],[]
    #读取data_sourse.json文件
    with open('data_source.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        #处理数据
        for item in data:
            text = item['k_qa_content']
            key,content = text.split('#\n')
            #添加到处理后列表中
            keywords.append(key)
            contents.append({"content":content})
    return keywords,contents

# #打印读取的数据
# print(len(processed_data))
# print(processed_data[1])
def api_embedding(texts,model_name):
    client = OpenAI(
api_key=os.environ['api_key'],  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)
    embeddings = []
    for input_text in texts:
        completion = client.embeddings.create(
            model=model_name,
            input=input_text,
            dimensions=64
    )
    embedding = completion.data[0].embedding
    embeddings.append(embedding)
    return embeddings

# 存储当前上传的文件内容
current_file_content = {"content": None}

def process_uploaded_file(file_obj):
    if file_obj is None:
        current_file_content["content"] = None
        return "未上传文件"
    try:
        # 获取文件扩展名
        file_extension = os.path.splitext(file_obj.name)[1].lower()
        
        # 处理图片文件
        if file_extension in ['.png', '.jpg', '.jpeg']:
            try:
                # 读取图片文件并转换为base64
                img_bytes = file_obj.read()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                current_file_content["content"] = f"data:image/{file_extension[1:]};base64,{img_base64}"
                return "图片上传成功，可以开始分析"
            except Exception as e:
                return f"图片处理失败: {str(e)}"
        
        # 处理JSON文件
        elif file_extension == '.json':
            try:
                with open(file_obj.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                current_file_content["content"] = content
                return content
            except Exception as e:
                return f"JSON文件读取失败: {str(e)}"
            
        else:
            return "不支持的文件格式"
            
    except Exception as e:
        current_file_content["content"] = None
        return f"文件处理失败: {str(e)}"

def llm_chat_with_file(message, history):
    client = ZhipuAI(api_key='59251781bb444688a4656508dceccaca.Lo7JWjD9N0XKfIu8')
    
    # 检查是否是请求分析文件
    if "分析文件" in message or "查看文件" in message or "解析文件" in message:
        if current_file_content["content"] is None:
            return "请先上传文件后再请求分析。"
        
        # 检查是否是图片内容
        if current_file_content["content"].startswith("data:image"):
            # 构造包含图片的消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"请帮我分析这张图片中的内容。用户的具体问题是：{message}"
                        },
                        {
                            "type": "image",
                            "image": current_file_content["content"]
                        }
                    ]
                }
            ]
            # 使用支持图片的模型
            model_name = "glm-4v-flash"
        else:
            # 处理普通文本内容
            messages = [
                {
                    "role": "user", 
                    "content": f"请帮我分析以下文件内容：\n{current_file_content['content']}\n\n用户的具体问题是：{message}"
                }
            ]
            # 使用普通对话模型
            model_name = "glm-4"
    else:
        # 普通对话
        messages = [{"role": "user", "content": message}]
        model_name = "glm-4"

    # 调用 API
    response = client.chat.completions.create(
        model=model_name,  # 根据内容类型选择合适的模型
        messages=messages,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # 创建gradio界面
    with gr.Blocks(fill_height=True) as demo:
        with gr.Row(scale=2):
            with gr.Column(scale=2):
                gr.Markdown("智能学习助教")
                chatbot = gr.ChatInterface(
                    fn=llm_chat_with_file,
                    title="智能学习助教",
                    multimodal=True, 
                    type='messages',
                    examples=['什么是人工智能'],
                    run_examples_on_click=False
                )
                gr.ClearButton(value='clear')
            
            with gr.Column(scale=1):
                gr.Markdown("## 文件上传")
                file_upload = gr.File(
                    label="上传文件", 
                    file_types=["image", ".json"],  # 添加对图片的支持
                    file_count="single"
                )
                file_content_display = gr.Textbox(
                    label="文件内容",
                    interactive=False
                )
                
                # 将事件绑定移动到这里
                file_upload.change(
                    fn=process_uploaded_file,
                    inputs=[file_upload],
                    outputs=[file_content_display]
                )
                
                gr.ClearButton(value='clear')

    demo.launch()

