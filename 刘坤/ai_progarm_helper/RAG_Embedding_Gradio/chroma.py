import json
import requests
import numpy as np
import chromadb

def get_embedding(text):
    """获取文本的向量"""
    url = "https://api.siliconflow.cn/v1/embeddings"
    data = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": text,
        "encoding_format": "float"
    }
    headers = {
        "Authorization": "Bearer sk-ozitaytcsoxjuiutcvkwutviquhayvxfthzoqipcipirgnnf",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        result = response.json()
        return result['data'][0]['embedding']
    except Exception as e:
        print(f"获取向量失败: {e}")
        return None

def load_knowledge_data():
    """读取知识库数据"""
    knowledge_data = []
    
    with open('data_source.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            text = item['k_qa_content']
            k, qa = text.split('#\n')
            knowledge_data.append({
                'k': k,
                'qa': qa
            })
    
    return knowledge_data

def init_database():
    """初始化向量数据库"""
    print("开始初始化向量数据库...")
    
    # 连接ChromaDB
    try:
        client = chromadb.HttpClient(host='localhost', port=8000)
        print("✓ 连接ChromaDB成功")
    except Exception as e:
        print(f"✗ 连接ChromaDB失败: {e}")
        return False
    
    # 删除旧集合
    collections = client.list_collections()
    collection_names = [col.name for col in collections]
    if "chroma_collection" in collection_names:
        client.delete_collection("chroma_collection")
        print("✓ 删除旧数据")
    
    # 创建新集合
    collection = client.create_collection("chroma_collection")
    print("✓ 创建新集合")
    
    # 加载知识库数据
    knowledge_data = load_knowledge_data()
    print(f"✓ 加载了 {len(knowledge_data)} 条知识")
    
    # 向量化并存储
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    
    print("开始向量化数据...")
    for i, item in enumerate(knowledge_data):
        print(f"处理 {i+1}/{len(knowledge_data)}: {item['k'][:50]}...")
        
        # 获取k的向量
        k_embedding = get_embedding(item['k'])
        if k_embedding:
            ids.append(str(i))
            documents.append(item['k'])
            metadatas.append({"qa": item['qa']})
            embeddings.append(k_embedding)
    
    # 批量添加到数据库
    if ids:
        try:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            print(f"✓ 成功存储 {len(ids)} 条向量数据")
            return True
        except Exception as e:
            print(f"✗ 存储数据失败: {e}")
            return False
    else:
        print("✗ 没有有效数据可存储")
        return False

if __name__ == '__main__':
    success = init_database()
    if success:
        print("\n🎉 数据库初始化完成！")
        print("现在可以运行 python main.py 启动聊天系统")
    else:
        print("\n❌ 数据库初始化失败")
        print("请检查ChromaDB服务和API配置")
    

    