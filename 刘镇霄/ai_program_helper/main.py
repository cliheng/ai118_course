import json
import numpy as np
import chromadb
import gradio as gr
from fuzzywuzzy import fuzz
from onnxruntime.transformers.models.stable_diffusion.benchmark import example_prompts

from AI_Class.Chroma向量数据库.考神助手.local_embedding import local_embeddings
from AI_Class.Chroma向量数据库.考神助手.qwen_ai import qwen3_ai_q
from embedding import embed_with_qwen


# JSON文件解析并拆分为关键字和答案以及切割
def json_parse(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    result = []
    chunk_size = 600
    chunk_overlap = 0
    sum = 0
    for item in data:
        sum = sum + 1
        k_qa_content = item['k_qa_content']
        keyword, answer = k_qa_content.split('#\n')
        print(f"正在切割中{sum}/{len(data)}...")
        for i in range(0, len(answer), chunk_size - chunk_overlap):
            chunk = answer[i:i + chunk_size]
            result.append([keyword, chunk])
    print("切割完成！")
    return result


# 查询相似度最高的答案
def search_similarity(query_keyword, collection):
    # 示例模糊查询关键词
    query_keyword = query_keyword
    query_results = collection.get(include=["metadatas"])
    best_match = None
    best_score = 0
    for metadata in query_results["metadatas"]:
        answer = metadata["answer"]
        score = fuzz.partial_ratio(query_keyword, answer)
        if score > best_score:
            best_score = score
            best_match = answer
    if best_match:
        # print("最佳匹配答案:", best_match)
        print("已最佳匹配答案！")
    else:
        print("未找到匹配答案。")
    queary = collection.query(query_embeddings=np.random.randn(1536), n_results=1)
    for result in queary['metadatas'][0]:
        # print(f"考神助手：{qwen3_ai_q(query_keyword,result['answer'])}")
        return qwen3_ai_q(query_keyword, result['answer'])


# 运行函数
def run(file_path, file_name):
    client = chromadb.HttpClient(host='localhost', port=8000)
    # client.delete_collection(name="chromadb_rag")
    collection = client.get_or_create_collection(name="chromadb_rag")
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
                examples=[["中序遍历"], ["换行符"], ["类和对象"]],
                title="学霸君考神系统",
                theme="soft",
                type = "messages"
            )
            demo.launch(share=True)
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
    run('assist/data_source.json', 'assist/data_source.json')
