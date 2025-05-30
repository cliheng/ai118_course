#打开json文件
import json
import chromadb
import os
import gradio as gr
from zhipuai import ZhipuAI
from openai import OpenAI
from dotenv import load_dotenv , find_dotenv
from modelscope import AutoTokenizer, AutoModel
import torch
import base64


def parse_json(file_path):
    #处理后数据存储list
    keywords,contents = [],[]
    # 读取data_source.json文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data_source = json.load(file)
        #处理数据
        for item in data_source:
            text = item['k_qa_content']
            key,content = text.split("#\n")
            #添加到处理后列表中
            keywords.append (key)
            contents.append ({"content": content})
            
    return keywords,contents

def api_embedding(texts,model_name):
    #多次循环中使用固定对象
    client = OpenAI(
        api_key=os.environ['api_key'],  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
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

def local_embedding(sentences):
    # tokenizer 输入文本转换模型输入需要变量类型
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    model.eval()

    # 输入文本转换模型输入类型
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # 生成Embedding
    with torch.no_grad():
        model_output = model(**encoded_input)
        # 从结果抽取模型生成Embedding
        sentence_embeddings = model_output[0][:,0]

    sentence_embeddings = sentence_embeddings.numpy().tolist()
    #print("Sentence embeddings:", len(sentence_embeddings))
    return sentence_embeddings

def chat_msg(message,history):
    #消息包含文件和文本
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    
        #处理图片代码
    img_text = "无图片文本"
    if len(message["files"]) > 0:
        img_path = message["files"][0]
        #提取图片中的文本
        img_text = fetch_text_from_image(img_path)
    
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})

        #RAG处理消息
        question = message["text"] + " " + img_text
        #向量化（批次为1的转换）
        q_emb = local_embedding([question])
        client = chromadb.HttpClient(host='localhost', port=8000)
        #chroma查询
        collection = client.get_collection("my_collection")
        result = collection.query(query_embeddings=q_emb, n_results=1)
        #提取结果中metadata
        content =""
        if len(result['metadatas']) > 0:
            content = result['metadatas'][0][0]['content']
        #提示
        prompt = f"你是一个精通python编程语言的专家,能依据参考资料来回答用户提出的各种问题和用户提交包含代码或错误信息的图片。用户问题:{message}\n\n 提交图片文本:{img_text}\n\n 输出格式:markdown格式 \n\n参考资料:{content}"
        
        #llm调用
        answer = llm_chat(prompt)

        #AI返回结果
        history.append({"role": "assistant", "content": answer.content})
        return history

def llm_chat(message):
    client = ZhipuAI(api_key=os.environ['ZHIPU_API_KEY'])
    response = client.chat.completions.create(
        model="glm-4-flash-250414",
        messages=[
            {"role": "user", "content": message}
        ]   
    )
    return response.choices[0].message
    return result

def fetch_text_from_image(image_path):
    with open(image_path, "rb") as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')
    
    client = ZhipuAI(api_key="b0d87913173c48018897755d559f6fe2.xHMxqIiXhVwpW9gi")
    response = client.chat.completions.create(
        model="glm-4v-plus-0111",
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
                    "text": "从图像中提取文本和代码"
                }
                ]
        }]
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
    collection = client.get_or_create_collection("my_collection")
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
    