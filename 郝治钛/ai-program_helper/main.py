import json
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from modelscope import AutoTokenizer, AutoModel
import torch
from zhipuai import ZhipuAI
import random
import chromadb
import gradio as gr
import base64
from zhipuai import ZhipuAI

# 解析 JSON 文件，提取 key 和 content
def parse_json(data_json):
    text_list = []
    #读取json文件
    with open (data_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # print(data)
        for item in data:
            contain = item['k_qa_content']
            key,content = contain.split('#\n')
            text_list.append({'keys':key,'content':content})
    return text_list  
# 获取文本的嵌入向量
def embedding(client,data_list,model,dimensions):
    completion = client.embeddings.create(
        model=model,
        input=data_list,
        dimensions=dimensions
    )
    return completion.data[0].embedding      
# 获取所有文本的嵌入向量
def get_embeddings():
    text_list = parse_json('data.json')
    load_dotenv(find_dotenv())  # 加载环境变量
    client = OpenAI(
        api_key=os.environ['api_key'],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    embeddings = []
    for sub_text in text_list:
        data_list = sub_text['keys']
        emb = embedding(client, data_list, 'text-embedding-v3', 768)
        embeddings.append(emb)
    return embeddings
# 将输入文本转换为本地模型的向量
def local_embedding(sentences):
     # tokenizer 输入文本转换模型输入需要变量类型
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-base-zh-v1.5')
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
# 调用 LLM 进行内容生成
def llm(content):
    
    client = ZhipuAI(api_key="93c6c03ecf874705b61af1679a69fa1e.pZXUMM6xddAarniz") # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-air-250414",  # 填写需要调用的模型编码
        messages=[
            {"role": "user", "content": f"{content}，根据资料做详细介绍"}
        ],
    )
    results = response.choices[0].message
    return results
# 添加数据到向量数据库并检索最相关内容
def random_response(message, image, history=None):
    client = chromadb.HttpClient(host='localhost', port=8000)
    embeddings = get_embeddings()
    id_len = len(embeddings)
    id_list = [str(i) for i in range(id_len)]
    data_list = parse_json('data.json')
    documents = [item['content'] for item in data_list]
    metadates = [{'keys': item['keys']} for item in data_list]
    collection = client.get_collection(name='data')
    collection.add(
        metadatas=metadates,
        embeddings=embeddings,
        ids=id_list,
        documents=documents
    )
    q_emb = local_embedding([message])
    result = collection.query(query_embeddings=q_emb, n_results=1)
    ref_content = result['documents'][0][0]
    # 构建 LLM 输入内容
    content = f"你是一个python专家,参考资料为\n\n{ref_content}，根据资料做详细介绍"
    # 处理图片
    img_base = None
    if image is not None and image != "":
        with open(image, 'rb') as img_file:
            img_base = base64.b64encode(img_file.read()).decode('utf-8')
    # 构建多模态 content
    multimodal_content = []
    if img_base:
        multimodal_content.append({
            "type": "image_url",
            "image_url": {"url": img_base}
        })
    multimodal_content.append({
        "type": "text",
        "text": content
    })
    # 调用 LLM
    client_llm = ZhipuAI(api_key="93c6c03ecf874705b61af1679a69fa1e.pZXUMM6xddAarniz")
    response = client_llm.chat.completions.create(
        model="glm-4v-plus-0111",
        messages=[{"role": "user", "content": multimodal_content}]
    )
    answer = response.choices[0].message.content
    return f"```markdown\n{answer}\n```"
# 创建图片识别
def gsimg(message, history):
    pass  # 保留空实现或后续扩展


if __name__ == '__main__':
    with gr.Blocks() as demo:
        # 创建聊天机器人组件，用于显示对话内容
        chatbot = gr.Chatbot()
        # 创建一个状态变量，用于存储对话历史
        state = gr.State([])
        # 创建一行布局，放置输入框和发送按钮
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="请输入内容...")
            img = gr.Image(type="filepath", label="上传图片")
            send_btn = gr.Button("发送")

        def user_send(message, image, history):
            response = random_response(message, image, history)
            history = history + [(f"{message}\n[图片: {image if image else '无'}]", response)]
            return history, "", None

        send_btn.click(user_send, inputs=[txt, img, state], outputs=[chatbot, txt, img])
        txt.submit(user_send, inputs=[txt, img, state], outputs=[chatbot, txt, img])

    demo.launch()