"""
此模块用于将处理好的数据存入chroma数据库
"""
import chromadb

# 将处理好的数据存储到向量数据库中
def savefile_chroma(embedding_list, doc_text, supplement, collection_name):
    client = chromadb.HttpClient(host="localhost")
    if collection_name not in [col.name for col in client.list_collections()]:
        collection = client.create_collection(collection_name)
        ids = []
        for i in range(len(embedding_list)):
            ids.append(f"id{i+1}")
        collection.add(
            ids=ids, embeddings=embedding_list, documents=doc_text, metadatas=supplement
        )
    else:
        collection = client.get_collection(collection_name)
        count = collection.count()
        existing_docs = collection.get(include=["documents"])["documents"]
        if existing_docs != doc_text:
            ids = []
            for index in range(len(embedding_list)):
                ids.append(f"id{index+1+count}")
            collection.upsert(
                ids=ids,
                embeddings=embedding_list,
                documents=doc_text,
                metadatas=supplement,
            )
