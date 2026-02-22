import os
import json
from openai import OpenAI
import gradio as gr
from embedding import process, api_embed
from chrome import chroma_insert, chroma_query
from ai_chat_bot import chat_with_model, question_embedding
from gradio_app import chat_gradio
import chromadb

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("emb_api_key"),  # 从环境变量中获取API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)

# 初始化ChromaDB客户端和集合
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_collection")

# 1. 数据预处理与向量化
keywords, concents = process("python1.json")  # 提取关键词和内容
embeddings = api_embed(keywords, client)  # 调用API生成嵌入

# 2. 插入数据到ChromaDB集合
chroma_insert(embeddings, concents, keywords, collection)
print("数据插入成功")

# 3. 定义问答主流程

def answer_func(message):
    # 用户问题向量化
    q_emb = question_embedding(message, client)
    # 查询ChromaDB获取最相关内容
    content = chroma_query(q_emb, collection, top_k=1)
    # 用大模型生成最终回答
    answer = chat_with_model(message, content)
    return answer

# 4. 启动Gradio界面
if __name__ == "__main__":
    chat_gradio(answer_func)