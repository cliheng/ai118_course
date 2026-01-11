import chromadb
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# 临时设置 API Key 到环境变量
os.environ["api_key"] = "sk-fc3b5e3ea6504dbea8928a79cfc7a46b"

def api_embedding(texts, model_name):
    client = OpenAI(
        api_key=os.environ["api_key"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    embeddings = []

    for input_text in texts:
        try:
            completion = client.embeddings.create(
                model=model_name,
                input=input_text,
                dimensions=768, 
            )
            embedding = completion.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            print(f"处理文本 {input_text} 时发生错误: {e}")
            # 可以选择跳过当前文本，或者采取其他处理方式
            continue
    return embeddings

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    # 示例文本
    sample_texts = ["这是一个示例文本", "这是另一个示例文本"]
    # 替换为实际支持的模型名称
    model_name = "text-embedding-v1"
    embedding = api_embedding(sample_texts, model_name)

    client = chromadb.HttpClient(host='localhost', port=8000)
    if not os.path.exists('chroma'):
        question = input('请输入问题：')
        q_emb = api_embedding([question], model_name) 
    collection = client.get_collection("my_collection")
    results = collection.query(query_embeddings=q_emb, n_results=1)
    content = ""
    if len(results['metadatas']) > 0:
        content = results['metadatas'][0][0]['content']
    prompt = f"你是一个精通python语言编程的专家，回答用户提出的问题。回答内容，请参考补充资料。\n\n补充资料：{content}"
    print(prompt)

    collection = client.get_or_create_collection("my_collection1")
    ids = []
    for i in range(len(embedding)):
        ids.append(f"id{i}")

    collection.add(
        ids=ids,
        embeddings=embedding,
        documents=sample_texts,
        metadatas=[{"content": text} for text in sample_texts],
    )

    print('向量化处理完成')