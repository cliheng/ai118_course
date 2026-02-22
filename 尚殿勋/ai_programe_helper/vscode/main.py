import chromadb
import json
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from modelscope import AutoModel, AutoTokenizer
import torch
import gradio as gr
from Flash import run_flash_demo  # 添加此行导入
# -*- coding: utf-8 -*-

def parse_json(file_path):
    keywords, contents = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        data_source = json.load(file)
        for item in data_source:
            text = item['k_qa_content']
            lines = text.split("\n")
            key = lines[0]
            content = "\n".join(lines[1:])
            keywords.append(key)
            contents.append({"content": content})
    return keywords, contents

def api_embedding(texts, model_name):
    client = OpenAI(
        api_key="sk-d8d6027eb4dc405ca00d37cb797c7d3b",
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

def local_embedding(sentences):
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
    model.eval()
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = sentence_embeddings.numpy().tolist()
    return sentence_embeddings

def llm_chat(messages, history=None):
    # 兼容Gradio ChatInterface的消息格式
    if isinstance(messages, list):
        user_input = messages[-1]["content"] if messages else ""
    else:
        user_input = messages
    # 检索知识库内容
    q_emb = local_embedding([user_input])
    collection = client.get_collection('example_collection')
    result = collection.query(query_embeddings=q_emb, n_results=1)
    content = ""
    if len(result['metadatas']) > 0:
        content = result['metadatas'][0][0]['content']
    # 拼接prompt
    prompt = f"你是一个精通python语言编程的专家，回答用户提出的问题。回答内容，请参考补充资料。\n\n补充资料:{content}\n\n用户问题:{user_input}"
    # LLM调用
    client_llm = OpenAI(
        api_key="sk-d8d6027eb4dc405ca00d37cb797c7d3b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client_llm.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content
if __name__ == "__main__":
    
    load_dotenv(find_dotenv())
    # Example usage
    keywords, contents = parse_json('data_source.json')
    # embeddings = api_embedding(keywords, "text-embedding-v3")  # 如果用API
    embeddings = local_embedding(keywords)  # 如果用本地模型

    client = chromadb.HttpClient(host='localhost', port=8000)
    # 如果已存在则用 get_collection，否则用 create_collection
    try:
        collection = client.create_collection(name='example_collection')
    except Exception:
        collection = client.get_collection(name='example_collection')

    ids = [f"id{i}" for i in range(len(embeddings))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=keywords,
        metadatas=contents
    )
    print('向量化处理完成')
        # 用户交互生成prompt
question = input('请输入问题：')
        # 向量化(批次为1的转换)
q_emb = local_embedding([question])
        # chroma查询
collection = client.get_collection('example_collection')
result = collection.query(query_embeddings=q_emb, n_results=1)
         # 提取结果中metadata
content = ""
if len(result['metadatas']) > 0:content = result['metadatas'][0][0]['content']
 
        # 提示
prompt = f"你是一个精通python语言编程的专家，回答用户提出的问题。回答内容，请参考补充资料。\n\n补充资料:{content}"

        # llm调用
answer = llm_chat(prompt)
print(answer)


# Gradio界面    
run_flash_demo()
gr.ChatInterface(
    llm_chat,
    type="messages",
).launch()








