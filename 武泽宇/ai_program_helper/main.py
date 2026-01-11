import json
import numpy as np
from embedding import embed_with_bge, embed_with_qwen
import os
import chromadb
import gradio as gr
from fuzzywuzzy import fuzz
from qwen3_AI import qwen3_ai_q



# 查询相似度最高的答案
def search_similarity(query_keyword, collection):
    # 获取集合中所有文档的元数据
    query_results = collection.get(include=["metadatas"])
    
    # 初始化最佳匹配和最高分变量
    best_match = None
    best_score = 0
    
    # 遍历所有元数据，使用模糊匹配算法查找最佳匹配答案
    for metadata in query_results["metadatas"]:
        answer = metadata["answer"]
        # 使用部分比率算法计算查询关键词与答案的相似度
        score = fuzz.partial_ratio(query_keyword, answer)
        
        # 更新最佳匹配结果
        if score > best_score:
            best_score = score
            best_match = answer
    
    # 输出匹配结果状态
    if best_match:
        print("已最佳匹配答案！")
    else:
        print("未找到匹配答案。")
    
    # 使用随机生成的嵌入向量查询集合获取一个结果
    queary = collection.query(query_embeddings=np.random.randn(1536), n_results=1)
    
    # 处理查询结果，使用qwen3模型生成最终回答
    for result in queary['metadatas'][0]:
        return qwen3_ai_q(query_keyword, result['answer'])

# JSON文件解析
def json_parse(file_path):
    # 以 UTF-8 编码打开指定路径的 JSON 文件，使用只读模式
    with open(file_path, 'r', encoding='utf-8') as file:
        # 从文件中加载 JSON 数据
        data = json.load(file)
    # 初始化一个空列表，用于存储最终处理结果
    result = []
    # 定义每个文本块的最大长度
    chunk_size = 600
    # 定义相邻文本块之间的重叠长度
    chunk_overlap = 0
    # 初始化计数器，用于记录当前处理的条目数量
    sum = 0
    # 遍历 JSON 数据中的每个条目
    for item in data:
        # 计数器加 1
        sum = sum + 1
        # 从当前条目中提取 k_qa_content 字段的值
        k_qa_content = item['k_qa_content']
        # 以 '#\n' 为分隔符，将 k_qa_content 拆分为关键词和答案两部分
        keyword, answer = k_qa_content.split('#\n')
        # 打印当前切割进度
        print(f"正在切割中{sum}/{len(data)}...")
        # 按指定的块大小和重叠长度，对答案文本进行分块处理
        for i in range(0, len(answer), chunk_size - chunk_overlap):
            # 从答案文本中截取一个文本块
            chunk = answer[i:i + chunk_size]
            # 将关键词和对应的文本块作为一个列表，添加到结果列表中
            result.append([keyword, chunk])
    # 打印切割完成的提示信息
    print("切割完成！")
    # 返回包含关键词和文本块对的结果列表
    return result

def run(file_path, file_name):
    client = chromadb.HttpClient(host='localhost', port=8000)
    # client.delete_collection(name="chromadb_rag")
    collection = client.get_or_create_collection(name="chromadb_rag")
    #如果数据库不为空，直接启动gradio
    if collection is not None and collection.count() > 0:
        def wrapper(query_keyword,history):
            return search_similarity(query_keyword, collection)
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 请输入你的问题，我来帮你")
            gr.ChatInterface(
                fn=wrapper,
                chatbot=gr.Chatbot(label="学霸君", height=550,type="messages"),
                examples=[["测试一"], ["测试二"], ["测试三"]],
                title="学霸君考神系统",
                theme="soft",
                type = "messages"
            )
            demo.launch()
    else:
        text_list = json_parse(file_path)
        keyword_list = []
        answers_list = []
        for text in text_list:
            keyword_list.append(text[0])
            answers_list.append({'answer': text[1]})
        # 使用闭源模型API
        print("开始向量化...")
        # 调用百炼模型向量化
        embedding_list = embed_with_qwen(keyword_list)
        # 调用本地模型向量化
        # embedding_list = local_embeddings(keyword_list)
        print(embedding_list)
        if type(embedding_list) == list:
            len_ids = len(text_list)
            ids_list = []
            for i in range(len_ids):
                ids_list.append(str(file_name) + str(i))
            collection.add(
                embeddings=embedding_list,
                documents=keyword_list,
                metadatas=answers_list,
                ids=ids_list
            )
            print("向量化完成！")
        else:
            return "error"
if __name__ == '__main__':
    # client = chromadb.HttpClient(host='localhost', port=8000, settings=chromadb.Settings(anonymized_telemetry=False))
    # client.delete_collection(name="chromadb-rag")
    file_path = "data_source.json"
    run(file_path,file_path)