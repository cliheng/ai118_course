import json
import chromadb
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from modelscope import AutoTokenizer, AutoModel 
import torch
import gradio as gr
import base64
from zhipuai import ZhipuAI

def parse_json(file_path):
    #处理后数据存储list
    keywords,contents = [],[]
    #读取数据源文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data_source = json.load(file)
        #遍历数据源文件
        for item in data_source:
            text = item['k_qa_content']
            key,content = text.split('#\n')
            #处理后的文本存储list
            keywords.append(key)
            contents.append({'content':content})
    return keywords,contents
def api_embedding(texts, model_name):
    client = OpenAI(
        api_key=os.environ["api_key"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    embeddings = []

    for input_text in texts:
        completion = client.embeddings.create(
            model=model_name,
            input=input_text,
            dimensions=768,
        )
        embedding = completion.data[0].embedding
        embeddings.append(embedding)
    return embeddings

# 全局只加载一次
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

def local_embedding(sentences):
    #终端运行chroma run --host 127.0.0.1 --port 8000
    # 直接使用全局变量
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:,0]
    return sentence_embeddings.numpy().tolist()
def chat_msg(message,history):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})

    img_text = "无图片文本"
    if len(message["files"]) > 0:
        img_path = message["files"][0]
        img_text = fetch_text_from_image(img_path)

    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})

        question = message["text"] + " " + img_text

        q_emb = local_embedding([question])
        client = chromadb.HttpClient(host='localhost', port=8000)
        collection = client.get_or_create_collection("my_collection1")  # 修改这里
        results = collection.query(query_embeddings=q_emb, n_results=1)

        content = ""
        # 安全判断，防止越界
        if results.get('metadatas') and len(results['metadatas']) > 0 and len(results['metadatas'][0]) > 0:
            content = results['metadatas'][0][0]['content']
        else:
            content = "未检索到相关知识内容。"

        prompt = f"你是一个精通编程语言的专家，依据补充资料回答用户提出的问题和用户提交包含代码或错误信息的图片。用户问题：{message} \n\n 提交图片文本：{img_text} \n\n 输出格式:markdown格式 \n\n 参考资料：{content}"

        answer = llm_chat(prompt)

        history.append({"role": "assistant", "content": answer.content})
        return history
def llm_chat(message):
    client = ZhipuAI(api_key=os.environ["ZHIPU_API_KEY"])
    response = client.chat.completions.create(
        model="glm-4-flash-250414",
        messages=[
            {"role": "user", "content": message}
        ]   
    )
    return response.choices[0].message
    # return result
def fetch_text_from_image(img_path):
    with open(img_path, "rb") as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')

    client = ZhipuAI(api_key=os.environ["ZHIPU_API_KEY"])
    response = client.chat.completions.create(
        model="glm-4v-plus-0111",
        messages=[
            {"role": "user", 
            "content":
            [
            {
                "type": "image_url",
                "image_url": 
                {
                    "url": img_base
                }
            },
            {
                "type": "text",
                "text": "从图像中提取文本和代码"
            }
            ] 
            },
        ],
    )
    return response.choices[0].message.content

with gr.Blocks() as demo: #容器
    # 创建一个聊天界面，用于显示聊天记录（维护history，识别不同角色）
    chatbot = gr.Chatbot(type = "messages")
    # 复合文本框，用于输入文本和图片
    tbox = gr.MultimodalTextbox(sources=['upload'], file_count="single", file_types=["image"])
    # 注册submit事件处理方法
    tbox.submit(fn=chat_msg, inputs=[tbox,chatbot], outputs=chatbot)
demo.launch()

if __name__ == "__main__":
    # 加载环境变量
    load_dotenv(find_dotenv())
    # 解析JSON文件
    keywords,contents = parse_json('data_source.json')
    # 使用API生成嵌入
    embedding = local_embedding(keywords)
    client = chromadb.HttpClient(host='localhost', port=8000)
    # 创建集合
    collection = client.get_or_create_collection("my_collection1")
    ids = []
    # 遍历生成的嵌入
    for i in range(len(embedding)):
        ids.append(f"id{i}")
    # 将嵌入添加到集合中
    collection.add(
        ids=ids,
        embeddings=embedding,
        documents=keywords,
        metadatas=contents,
    )
    print('向量化处理完成')