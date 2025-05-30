import chromadb

# 初始化ChromaDB客户端 
client = chromadb.Client()

# 2. 创建或获取集合（collection）
collection = client.get_or_create_collection(name="my_collection")
# 3. 插入数据到集合中


def chroma_insert(embeddings, concents, keywords, collection):
    """根据embedding模块得到的embeddings、concents和keywords插入数据到ChromaDB集合中
    Args:
        embeddings (list): 嵌入向量列表
        concents (list): 内容列表
        keywords (list): 关键词列表
        collection: ChromaDB集合对象
        Returns:collection: 插入数据后的ChromaDB集合对象
    """
    ids = [f"id_{i}" for i in range(len(embeddings))]
    dict_keywords = [{"keyword": item} for item in keywords]
    collection.add(
        embeddings=embeddings,
        documents=concents,
        metadatas=dict_keywords,
        ids=ids
    )
    return collection


def chroma_query(query_embedding, collection, top_k=1):
    """根据查询向量从ChromaDB集合中查询最相似的内容
    input:
        query_embedding (list): 查询向量
        collection (chromadb.Collection): ChromaDB集合对象
        top_k (int): 返回的最相似结果数量
    output:
        date (str): 返回的最相似内容   
    """
    result = collection.query(
        query_embeddings=[query_embedding],  # 需要二维列表
        n_results=top_k
    )
    if result["documents"] and result["documents"][0]:
        date = result["documents"][0][0]
    else:
        date = "未找到相关内容"
    return date
