"""
此模块用于实现RAG的向量检索
"""

import chromadb

client = chromadb.HttpClient(host="localhost")


def search(query, collection_name):
    collection = client.get_collection(collection_name)
    result = collection.query(query_embeddings=query, n_results=1)

    content = ""
    if result["metadatas"] is not None and len(result["metadatas"]) > 0:
        content = result["metadatas"][0][0]["content"]

    return content
