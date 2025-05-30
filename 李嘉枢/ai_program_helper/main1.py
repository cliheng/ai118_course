import os.path as path
import os
from openai import OpenAI
from zhipuai import ZhipuAI
import json
from dotenv import load_dotenv, find_dotenv

from modelscope import AutoTokenizer, AutoModel
import torch
import chromadb # 导入 chromadb 库
import gradio as gr # 导入 gradio 库
import base64
def parse_json(file_path):
    keywords,contents= [] ,[] # 存储关键词
    with open(file_path, 'r', encoding='utf-8') as file:  # 使用传入的 file_path 参数
        data_source = json.load(file)  # 加载原始数据
    # 处理数据
    for item in data_source:# 提取文本内容（假设原始数据中每个条目有 'k_qa_content' 字段）
        text = item.get('k_qa_content', '')  # 避免 KeyError（若字段不存在则返回空字符串）
        key, content = text.split("#\n") 
        keywords.append(key)  # 存储关键词
        contents.append({'content': content})# 存储内容
    return keywords,contents  # 返回处理后的列表，确保函数返回数据
## 调用api函数
def api_embedding(input_text, client, model_name):
    completion = client.embeddings.create(
        model=model_name,
        input=input_text,       
        dimensions=768          
    )
    # 提取嵌入向量
    embedding = completion.data[0].embedding
    return embedding
## 本地嵌入函数
def local_embedding(sentences):
#  加载模型
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    model.eval()
#  输入文本
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#  计算文本嵌入
    with torch.no_grad():#  使用 torch.no_grad() 禁用梯度计算
        model_output = model(**encoded_input)#  计算模型输出
    #  提取句子嵌入
        sentence_embeddings = model_output[0][:, 0]#  取第一个token的嵌入向量
    sentence_embeddings = sentence_embeddings.numpy().tolist ()  # 转换为列表格式
    return sentence_embeddings



def llm_chat(message):
        # 初始化ZhipuAI客户端
    client= ZhipuAI(api_key=os.environ['zhipuai_api_key'])
    response = client.chat.completions.create(
        model="glm-4-flash-250414",  # 模型名称
        messages=[{
                "role": "user","content": message}]
 )
    result=response.choices[0].message.content
    return result

def fetch_text_from_image(image_path):
    with open(image_path, 'rb') as image_file:
        img_base = base64.b64encode(image_file.read()).decode('utf-8')

    client = ZhipuAI(api_key="14accb92f1304b9683fc1150a29ebb8d.HLlP4tMZT2Ti6hPO")  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4v-plus-0111",  # 填写需要调用的模型名称
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
                        "text": "请提取图片中的文本内容，并将其作为回答的一部分。"
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    # 初始化客户端（类名需与实际库匹配）

    client = chromadb.HttpClient(host='localhost', port=8000)

    if not os.path.exists('127.0.0.1'):
            
        keywords,contents = parse_json('data_source.json')
    
        embeddings = local_embedding(keywords)
        #print(len(embeddings))
        
        #创建数据库
        collection = client.get_or_create_collection("my_collection")

        # 添加向量数据
        ids = []#  存储ID
        for  i in range(len(keywords)):
            ids.append(f"id{i+1}")# 生成唯一ID
        
        collection.add(
        ids=ids,  # 关联的ID
        embeddings=embeddings,#context关联的干化词
        metadatas=contents,  # 关联的内容
        documents=keywords, 
        )
        print("向量数据库已创建，包含以下数据：")
    #用户交互生成prompt
    def chat_msg(message, history):
        #消息包含文件和文本
        for x in message["files"]:
            history.append({"role": "user", "content": {"path": x}})
        
        img_text = "无图片文本"   
        
        if len(message["files"]) > 0:
            img_path=message["files"][0]
            #提取图片中的文本
            img_text = fetch_text_from_image(img_path)
        
        if message['text'] is not None:
            history.append({"role": "user", "content": message['text']})                   
        
        #RAG处理消息
        question=message['text']+" "+img_text #合并用户问题和图片文本
        #向量化(批次为1的转换)
        q_emb=local_embedding([question]) 
        #chroma查询
        collection=client.get_collection("my_collection")
        result=collection.query(
        query_embeddings=q_emb,  # 查询向量
        n_results=1  # 返回最相似的1个结果
        )   
        content = ""# 提取结果中metadata的内容
        #提取结果中metadata的内容
        if len(result['metadatas']) > 0:
            content= result['metadatas'][0][0]['content']
            #提示
        prompt=f"你是一个精通于python的AI助手，回答用户提出的问题和用户提交的包含代码活错误信息的照片，尽量用中文回答。用户问题：{message} \n\n 提交图片文本：{img_text}\n\n 输出格式：markdown文本\n\n 补充资料:{content}"
        
        #llm调用
        answer=llm_chat(prompt)
        
        #返回结果
        history.append({"role": "assistant", "content": answer})
        return history
    #Gradio界面
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type="messages")
        tbox = gr.MultimodalTextbox(sources="upload", file_count="single", file_types=["image"])
        tbox.submit(fn=chat_msg, inputs=[tbox,chatbot], outputs=chatbot)
    demo.launch()