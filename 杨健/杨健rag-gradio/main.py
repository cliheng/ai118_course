import json
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import chromadb
from zhipuai import ZhipuAI
import gradio as gr
import base64

def pare_json(file_path):   #解析json文件,并返回列表嵌套字典格式数据
    keywords,contents = [],[]
    with open(file_path,"r",encoding = "utf-8") as file:
        data_source = json.load(file)

        for item in data_source:
            text = item["k_qa_content"]
            key,content = text.split('#\n')
            keywords.append(key)
            contents.append({'content':content})
    
    return keywords,contents

def api_embedding(input_text,client): #调用embedding的api的函数
    completion = client.embeddings.create(
        model="text-embedding-v2",
        input=input_text,
        dimensions=768,
        encoding_format="float"# 可选值：float，base64 
        )
    embedding = completion.data[0].embedding
    return embedding

openai_client = OpenAI(
    api_key=os.environ['api_key'],  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1") # 百炼服务的base_url
embeddings = []
keywords,contents= pare_json("data_source.json")
for sub_dict in keywords:
    embedding = api_embedding(sub_dict,openai_client)
    embeddings.append(embedding)       # 把干化词向量化了

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection("my_collection")
if not os.path.exists("chroma"): # 向量入库
    
    #添加向量数据
    ids = []
    for i in range(len(embeddings)):
        ids.append(f'id{i}')

    collection.add(
        ids=ids,
        embeddings=embeddings, # context关联的embeddings(干化词)
        documents=keywords, # 原始文本
        metadatas=contents, # 原始文本的元数据信息
    )


def llm_chat(message,history=None):
    client = ZhipuAI(api_key=os.environ['zhipu_key'])  # 请填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-flash-250414",  # 请填写您要调用的模型名称
        messages=[
            {"role": "user", "content": message}] 
    )

    return response.choices[0].message.content

def chat_msg(message,history):

    for x in message['files']:
        history.append({"role": "user", "content": {'path':x}})

    img_text = "无图片文本"
    if len(message['files']) > 0:
        img_path = message['files'][0]
        img_text = fetch_text_from_image(img_path)

    if message.get('image') is not None:
        history.append({"role": "user", "content": {'image':message['image']}})
    if message.get('text') is not None:
        history.append({"role": "user", "content": message['text']})

    
    client = chromadb.HttpClient(host="localhost", port=8000)
    # 用户交互生成prompt
    question = message['text'] + '' + img_text
    # 向量化（批次为1）
    q_emb = api_embedding([question],openai_client)
    # chroma查询
    collection = chroma_client.get_collection("my_collection")
    result = collection.query(
        query_embeddings=q_emb, # 向量化后的问题
        n_results=1, # 返回的结果数量
    )
    #提取记过中metadata
    content= ""
    if len(result['metadatas'][0]) > 0: # 检查是否存在元数据
        content = result['metadatas'][0][0]['content'] # 提取元数据中的content字段
    
    prompt = f"你是一个精通python语言变成的专家，能依据参考资料来回答用户提出的各种问题和用户提交包含代码或错误信息的图片。用户的问题：{message} \n\n 提交图片文本：{img_text} \n\n 输出格式：marksown\n\n参考资料：{content}"

    answer = llm_chat(prompt)
    
    history.append({"role": "assistant", "content": answer})
    return history

def fetch_text_from_image(image_path):
    with open(image_path, "rb") as image_file:
        img_base = base64.b64encode(image_file.read()).decode("utf-8")

    client = ZhipuAI(api_key=os.environ['zhipu_key'])  # 请填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4v-plus-0111",  # 请填写您要调用的模型名称
        messages=[
            {"role": "user", "content":[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_base
                    }
                },
                {
                    "type": "text",
                    "text": "从图像中提取文本和代码"
                }
            ]
            }]
    )
    return response.choices[0].message.content



openai_client = OpenAI(
    api_key=os.environ['api_key'],  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")  # 百炼服务的base_url

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    tbox = gr.MultimodalTextbox(sources=['upload'],file_count='single',file_types=['image'])
    tbox.submit(fn=chat_msg,inputs=[tbox,chatbot],outputs=chatbot)
demo.launch()
