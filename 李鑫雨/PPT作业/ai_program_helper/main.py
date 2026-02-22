import json
import os
import zhipuai
import chromadb
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from modelscope import AutoTokenizer, AutoModel
import torch
import gradio as gr
import base64


def parse_json(file_path):
    #处理数据   
    processed_data = []
    keywords,contents = [],[]
    #读取数据
    with open(file_path, 'r', encoding='utf-8') as file:
        data_source = json.load(file)
        #处理数据
        for item in data_source:
            text = item["k_qa_content"]
            key, content = text.split("#\n")
            keywords.append(key)
            contents.append({"content":content})
        return keywords, contents

def api_embedding(input_text, client):
    completion = client.embeddings.create(
        model="text-embedding-v2",
        input=input_text,
        dimensions=768,
    )
    embedding = completion.data[0].embedding
    return embedding

def load_embeddings(sentences):

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    model.eval()

    # 编码输入
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # 获取嵌入向量
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    
    # normalize embeddings
    sentence_embeddings = sentence_embeddings.numpy().tolist()
    
    return sentence_embeddings
def llm_chat(message):
    api_key = os.getenv('ZHIPUAI_API_KEY')
    if not api_key:
        raise ValueError("未找到 ZHIPUAI_API_KEY 环境变量")
        
    client = zhipuai.ZhipuAI(api_key=api_key)
    response = client.chat.completions.create(
        model="glm-4-flash-250414",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message

# 全局变量
client = None
collection = None

def chat_msg(message, history):
    global client, collection
    
    if history is None:
        history = []
    
    try:
        # 处理文件上传
        img_text = '无文本文件'
        if message.get("files") and len(message["files"]) > 0:
            img_path = message["files"][0]
            try:
                # 暂时跳过图片处理
                img_text = "图片处理功能待实现"
            except Exception as e:
                print(f"图片处理错误: {str(e)}")
                img_text = "图片处理失败"        # 处理文本消息
        if message.get("text"):
            history.append({"role": "user", "content": message["text"]})
            
            try:
                # 构建查询
                question = message["text"] + " " + img_text
                q_emb = load_embeddings([question])
                
                # 查询向量数据库
                if collection is None:
                    collection = client.get_collection("my_collection")   
                results = collection.query(
                    query_embeddings=q_emb, 
                    n_results=1
                )
                
                # 提取内容
                retrieved_content = ""
                if results['metadatas'] and len(results['metadatas'][0]) > 0:
                    retrieved_content = results['metadatas'][0][0].get('content', '')

                # 构建提示词
                prompt = f"""你是一个精通python语言编程的专家,能依据参考资料来回答用户提出的各种问题和用户提交包含代码或错误信息的图片。
用户问题: {message["text"]}
提交图片文本: {img_text}
输出格式: markdown格式
参考资料: {retrieved_content}"""

                # LLM调用
                answer = llm_chat(prompt)
                history.append({"role": "assistant", "content": answer.content})
            except Exception as e:
                error_message = f"处理消息时出错: {str(e)}"
                history.append({"role": "assistant", "content": error_message})
    except Exception as e:
        error_message = f"系统错误: {str(e)}"
        if history:
            history.append({"role": "assistant", "content": error_message})
        else:
            history = [{"role": "assistant", "content": error_message}]
    
    return history

if __name__ == "__main__":
    # 加载环境变量
    load_dotenv(find_dotenv())
    
    # 初始化向量数据库
    client = chromadb.PersistentClient(path="./db")
    
    # 如果数据库不存在，则创建并导入数据
    if not os.path.exists('db'):
        keywords, contents = parse_json('data_source.json')
        embeddings = load_embeddings(keywords)
        collection = client.get_or_create_collection("my_collection")
        
        ids = [f"id{i}" for i in range(len(embeddings))]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=keywords,
            metadatas=contents
        )
    
    # 创建Gradio界面
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type="messages")
        tbox = gr.MultimodalTextbox(
            sources=['upload'],
            file_count="single",
            file_types=["image"]
        )
        tbox.submit(fn=chat_msg, inputs=[tbox, chatbot], outputs=chatbot)
    
    demo.launch()