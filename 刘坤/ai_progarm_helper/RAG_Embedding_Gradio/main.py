import gradio as gr
import openai
from openai import OpenAI
import requests
import json
import chromadb
import numpy as np

# API配置
openai_client = OpenAI(
    api_key="7hXKam!8%4(3Dv9X&b3LF&J",
    base_url="http://142.171.230.23:54638/v1"
)

def check_database():
    """检查数据库是否已初始化"""
    try:
        chroma_client = chromadb.HttpClient(host='localhost', port=8000)
        collection = chroma_client.get_collection("chroma_collection")
        count = collection.count()
        if count > 0:
            print(f"✓ 数据库已初始化，包含 {count} 条数据")
            return collection
        else:
            print("⚠ 数据库为空，请先运行 python chroma.py 初始化数据库")
            return None
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        print("请确保ChromaDB服务运行在localhost:8000，并先运行 python chroma.py")
        return None

# 连接ChromaDB
collection = check_database()

def get_embedding(text):
    """获取文本的向量"""
    url = "https://api.siliconflow.cn/v1/embeddings"
    data = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": text,
        "encoding_format": "float"
    }
    headers = {
        "Authorization": "Bearer sk-ozitaytcsoxjuiutcvkwutviquhayvxfthzoqipcipirgnnf",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        result = response.json()
        return result['data'][0]['embedding']
    except:
        return None

def search_knowledge(question):
    """在知识库中搜索相关内容"""
    if not collection:
        return None
    
    # 获取问题的向量
    question_vector = get_embedding(question)
    if not question_vector:
        return None
    
    # 搜索相似内容
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=1
    )
    
    if results['metadatas'] and len(results['metadatas'][0]) > 0:
        return results['metadatas'][0][0]['qa']
    return None

def answer_question(question):
    """回答用户问题"""
    # 搜索知识库
    knowledge = search_knowledge(question)
    
    # 构建提示
    if knowledge:
        prompt = f"问题：{question}\n参考资料：{knowledge}\n请根据参考资料回答问题。"
    else:
        prompt = f"问题：{question}\n请回答这个问题。"
    
    # 调用AI模型
    try:
        response = openai_client.chat.completions.create(
            model="gemini-2.5-pro-exp-03-25",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"出错了：{str(e)}"

def chat(message, history):
    """聊天函数"""
    if message.strip():
        answer = answer_question(message)
        # 使用简单的列表格式
        history.append([message, answer])
    return history, ""

# 创建简单界面
def create_interface():
    with gr.Blocks(title="RAG聊天系统") as demo:
        if collection:
            gr.Markdown("# 📚 RAG聊天系统\n这是一个简单的问答系统，可以根据知识库回答问题。")
        else:
            gr.Markdown("# ⚠️ 数据库未初始化\n请先运行 `python chroma.py` 初始化数据库。")
        
        chatbot = gr.Chatbot(label="对话", height=400)
        
        with gr.Row():
            msg = gr.Textbox(
                label="输入问题", 
                placeholder="请输入你的问题..." if collection else "请先初始化数据库...",
                interactive=bool(collection)
            )
            send_btn = gr.Button("发送", interactive=bool(collection))
        
        clear_btn = gr.Button("清空对话")
        
        # 绑定事件
        if collection:
            send_btn.click(chat, [msg, chatbot], [chatbot, msg])
            msg.submit(chat, [msg, chatbot], [chatbot, msg])
        
        clear_btn.click(lambda: [], outputs=chatbot)
    
    return demo

if __name__ == "__main__":
    if collection:
        print("启动聊天系统...")
        demo = create_interface()
        demo.launch(share=True, debug=True)
    else:
        print("数据库未初始化，请先运行 python chroma.py")
        print("初始化完成后再运行 python main.py")
