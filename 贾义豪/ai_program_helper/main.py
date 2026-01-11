import json  # 导入json模块，用于处理json数据
import os  # 导入os模块，用于环境变量等操作
from openai import OpenAI  # 导入OpenAI库，用于调用OpenAI接口
from dotenv import load_dotenv, find_dotenv  # 导入dotenv，用于加载.env环境变量
import torch  # 导入PyTorch，用于深度学习模型推理
from transformers import AutoTokenizer, AutoModel  # 导入transformers库，用于加载预训练模型和分词器
import chromadb  # 导入ChromaDB库，用于向量数据库操作
from zhipuai import ZhipuAI  # 导入智谱AI SDK
import gradio as gr  # 导入Gradio，用于快速搭建Web界面
import base64  # 导入base64，用于图片编码

# 解析 data_source.json，提取关键词和内容
def parse_json(data_source):
    keywords, contents = [], []  # 初始化关键词和内容列表
    with open('data_source.json', 'r', encoding='utf-8') as file:  # 打开json文件
        data_source = json.load(file)  # 加载json数据
        for item in data_source:  # 遍历每一项
            text = item['k_qa_content']  # 取出内容字段
            key, content = text.split('#\n')  # 按分隔符分割为关键词和内容
            keywords.append(key)  # 添加关键词
            contents.append({'content': content})  # 添加内容
    return keywords, contents  # 返回关键词和内容

# 调用 OpenAI embedding 接口，将文本转为向量
def api_embedding(input_text, client, model_name):  
    completion = client.embeddings.create(
        model=model_name,  # 指定模型名
        input=input_text,  # 输入文本
        dimensions=600,  # 指定向量维度
    )
    embedding = completion.data[0].embedding  # 取出第一个embedding向量
    return embedding  # 返回向量

# 使用本地模型将文本转为向量
def local_embedding(sentences):
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')  # 加载分词器
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')  # 加载模型
    model.eval()  # 设置为推理模式
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')  # 分词并转为张量
    with torch.no_grad():  # 关闭梯度计算，加速推理
        model_output = model(**encoded_input)  # 前向推理
        sentence_embeddings = model_output[0][:, 0]  # 取每个句子的[CLS]向量
    sentence_embeddings = sentence_embeddings.numpy().tolist()  # 转为列表
    return sentence_embeddings  # 返回向量

# 调用智谱AI大模型进行对话
def llm_chat(message):
    client = ZhipuAI(api_key=os.environ['zhipu_key'])  # 初始化智谱AI客户端
    response = client.chat.completions.create(
        model="glm-4v-plus-0111",  # 指定模型
        messages=[
            {
                "role": "user",  # 用户角色
                "content": message  # 用户输入内容
            }
        ]
    )
    # 取出内容
    return response.choices[0].message.content  # 返回模型回复内容

# 处理图片，调用智谱AI多模态接口提取图片内容
def fetch_from_image(image_path): 
    with open(image_path, 'rb') as img_file:  # 以二进制方式打开图片
        img_base = base64.b64encode(img_file.read()).decode('utf-8')  # 编码为base64字符串
    client = ZhipuAI(api_key=os.environ['zhipu_key'])  # 初始化智谱AI客户端
    response = client.chat.completions.create(
        model="glm-4v-plus-0111",  # 指定模型
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",  # 指定内容类型为图片
                        "image_url": {
                            "url": img_base  # base64图片内容
                        }
                    },
                    {
                        "type": "text",  # 额外的文本提示
                        "text": "从图像中提取文本和代码"
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content  # 返回提取的内容

# 聊天主逻辑，处理文本和图片输入
def chat_msg(messages, history):
    img_text = '无图片内容'  # 默认图片内容
    files = messages.get("files", [])  # 获取上传的文件列表
    for x in files:
        history.append({"role": "user", "content": {"path": x}})  # 记录图片路径到历史
    if len(files) > 0:
        img_path = files[0]  # 取第一张图片
        img_text = fetch_from_image(img_path)  # 提取图片内容
    # 处理文本消息
    if messages.get("text"):
        history.append({"role": "user", "content": messages["text"]})  # 记录文本到历史
        question = messages["text"] + '' + img_text  # 拼接文本和图片内容
        q_emb = local_embedding([question])  # 生成查询向量
        collection = client.get_or_create_collection(name="my_collection")  # 获取或创建向量集合
        results = collection.query(
            query_embeddings=q_emb,  # 查询向量
            n_results=1  # 返回最相似的1条
        )
        contents = ''
        # 取出最相关的内容
        if results.get('metadatas') and len(results['metadatas']) > 0 and len(results['metadatas'][0]) > 0:
            contents = results['metadatas'][0][0].get('content', '')
        # 构造prompt
        prompt = f'你是一个精通python的专家，回答用户的问题和用户提交的包含代码和错误的图片:{messages}，\n\n 提交的图片 {img_text} \n\n回答格式markdown\\n\n 回答内容参考{contents}'
        answer = llm_chat(prompt)  # 调用大模型生成回复
        history.append({"role": "assistant", "content": answer})  # 记录回复到历史
    return history  # 返回历史

if __name__ == '__main__':
    load_dotenv(find_dotenv())  # 加载环境变量
    client = chromadb.HttpClient(host="localhost", port=8000)  # 连接 ChromaDB 向量数据库

    with gr.Blocks() as demo:  # 创建Gradio界面
        chatbot = gr.Chatbot(type="messages")  # 聊天窗口
        tbox = gr.MultimodalTextbox(sources=['upload'], file_count="single", file_types=["image"])  # 支持图片上传的输入框
        tbox.submit(fn=chat_msg, inputs=[tbox, chatbot], outputs=chatbot)  # 提交时调用chat_msg
        demo.launch()  # 启动Web服务
