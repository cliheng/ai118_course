import json
import chromadb
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from modelscope import AutoTokenizer, AutoModel
import torch  # pytorch包(AI模型搭建、训练、运行)所有数据类型Tensor(张量)
import gradio as gr

# 解释json文件方法
def parse_json(file_path):
    # 处理后数据存储list
    keywords, contents = [], []
    # 读取data_source.json文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data_source = json.load(file)
        # 处理数据
        for item in data_source:
            text = item['k_qa_content']
            key, content = text.split("#\n")
            # 添加到处理后列表中
            keywords.append(key)
            contents.append({"content": content})
    return keywords, contents


# 多次循环中使用固定对象
def api_embedding(texts, model_name):
    client = OpenAI(
        api_key=os.environ['api_key'],  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )
    embeddings = []
    for input_text in texts:
        completion = client.embeddings.create(
            model=model_name,
            input=input_text,
            dimensions=768
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
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
    # 生成Embedding
    with torch.no_grad():
        model_output = model(**encoded_input)
        # 从结果抽取模型生成Embedding
        sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = sentence_embeddings.numpy().tolist()
    # print("Sentence embeddings:", len(sentence_embeddings))
    return sentence_embeddings

def llm_chat(prompt):
    client = OpenAI(
        api_key="sk-d8d6027eb4dc405ca00d37cb797c7d3b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client.chat.completions.create(
        model="qwen-turbo",  # 你可以换成你实际的模型名
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message
if __name__ == '__main__':
    load_dotenv(find_dotenv())
    keywords, contents = parse_json('data_source.json')
    embeddings = local_embedding(keywords)
    # embeddings = api_embedding(keywords, "text-embedding-v2")
    # print(len(embeddings))
    # 创建向量数据库连接，并返回数据库对象（启动chromadb服务）
    client = chromadb.HttpClient(host='localhost', port=8000)
    # 直接使用 get_or_create_collection 方法
    collection = client.get_or_create_collection('my_collection')
    # 添加向量数据
    ids = []
    for i in range(len(embeddings)):
        ids.append(f"id{i}")
    collection.add(
        ids=ids,
        embeddings=embeddings,  # context关联的embedding（干化词）
        documents=keywords,
        metadatas=contents
    )
    print('向量化处理完成!')
    if not os.path.exists('chroma'):
        os.makedirs('chroma')
def chat(message, history):

    # 用户交互生成 prompt
    question = message
    content = ""
    try:
        # 向量化（批次为1的转换）
        q_emb = local_embedding([question])
        # chroma查询
        collection = client.get_collection('my_collection')
        result = collection.query(
            query_embeddings=q_emb,
            n_results=1 
        )
        # 提取结果中metadata
        if result['metadatas'] and result['metadatas'][0]:
            content = result['metadatas'][0][0]['content']
    except Exception as e:
        print(f"查询时发生错误: {e}")

    # 构建提示
    prompt = f"你是一个精通python语言编程的专家，依据参考资料来回答用户提出的各种问题。问题:{message}。回答内容，请参考补充资料。\n\n输出格式:markdown格式\n\n补充资料:{content}"
    try:
        # llm调用
        answer = llm_chat(prompt)
        return answer.content
    except Exception as e:
        print(f"LLM调用时发生错误: {e}")
        return "抱歉，处理您的请求时发生了错误。"

    # 返回回答内容
    return answer.content


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    textbox = gr.Textbox()
    textbox.submit(lambda x: [(x, chat(x, []))], textbox, chatbot)
demo.launch()


gr.ChatInterface(
    fn=chat, 
    type="messages"
).launch()
