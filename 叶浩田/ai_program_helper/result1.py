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

# 初始化全局变量
client = None
collection = None

# 解释json文件方法
def parse_json(file_path):
    keywords, contents = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        data_source = json.load(file)
        for item in data_source:
            text = item['k_qa_content']
            key, content = text.split("#\n")
            keywords.append(key)
            contents.append({"content": content})
    return keywords, contents

# 本地嵌入模型
def local_embedding(sentences):
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    model.eval()
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
    return sentence_embeddings.numpy().tolist()

# LLM聊天函数
def llm_chat(prompt):
    client = OpenAI(
        api_key="sk-d8d6027eb4dc405ca00d37cb797c7d3b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message

# 初始化向量数据库
def init_vector_db():
    global client, collection
    load_dotenv(find_dotenv())
    keywords, contents = parse_json('data_source.json')
    embeddings = local_embedding(keywords)
    
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_or_create_collection('my_collection')
    
    ids = [f"id{i}" for i in range(len(embeddings))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=keywords,
        metadatas=contents
    )
    print('向量化处理完成!')
    if not os.path.exists('chroma'):
        os.makedirs('chroma')

# 文本聊天处理
def text_chat(message, history):
    content = ""
    try:
        q_emb = local_embedding([message])
        result = collection.query(
            query_embeddings=q_emb,
            n_results=1
        )
        if result['metadatas'] and result['metadatas'][0]:
            content = result['metadatas'][0][0]['content']
    except Exception as e:
        print(f"查询时发生错误: {e}")

    prompt = f"你是一个精通python语言编程的专家，依据参考资料来回答用户提出的各种问题。问题:{message}。回答内容，请参考补充资料。\n\n输出格式:markdown格式\n\n补充资料:{content}"
    try:
        answer = llm_chat(prompt)
        return answer.content
    except Exception as e:
        print(f"LLM调用时发生错误: {e}")
        return "抱歉，处理您的请求时发生了错误。"

# 图文问答处理
def image_chat(message, files):
    try:
        zhipu_client = ZhipuAI(api_key="04d2bda59ba34e2388ad33c3541a9e65.4meytIG8TRAS12pC")
        
        if files:
            img_path = files if isinstance(files, str) else files.name if hasattr(files, 'name') else files
            if not os.path.exists(img_path):
                return "无法访问图片文件"
                
            with open(img_path, 'rb') as img_file:
                img_base = base64.b64encode(img_file.read()).decode('utf-8')
                
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base}"}
                },
                {"type": "text", "text": message}
            ]
            
            response = zhipu_client.chat.completions.create(
                model="glm-4v-plus-0111",
                messages=[{"role": "user", "content": content}]
            )
            return response.choices[0].message.content
        else:
            return text_chat(message, [])
    except Exception as e:
        return f"处理过程中出现错误：{str(e)}"

# 创建Gradio界面
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 多功能问答系统")
        with gr.Tabs():
            with gr.TabItem("文本问答"):
                gr.ChatInterface(
                    fn=text_chat,
                    title="文本问答",
                    description="输入您的问题，系统将基于知识库回答"
                )
            
            with gr.TabItem("图文问答"):
                with gr.Row():
                    file_output = gr.File(
                        label="上传图片",
                        file_types=["image"],
                        type="filepath"
                    )
                    text_input = gr.Textbox(label="输入问题")
                
                text_output = gr.Textbox(label="回答")
                submit_btn = gr.Button("提交")
                
                submit_btn.click(
                    fn=image_chat,
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
    
    return demo

if __name__ == '__main__':
    init_vector_db()
    demo = create_interface()
    demo.launch(share=True)