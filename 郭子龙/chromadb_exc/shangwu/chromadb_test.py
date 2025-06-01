import chromadb
import numpy as np
if __name__ == "__main__":
    #创建一个chroma客户端
    client = chromadb.HttpClient(host='localhost', port=8000)  


    # 创建chroma数据库
    # db = client.create_collection('my_collection', metadata={"hnsw:space": "l2"})   #2范式collection


#     collection = client.create_collection(
#     name="collection_name",
#     metadata={"hnsw:space": "cosine"} # l2 is the default    #cosine 余弦相似度collecttion
# )

    collection = client.get_collection(name="collection_name")  # 获取已存在的collection

    # 列举出创建的collection
    collections = client.list_collections()
    print(collections)


    
    collection.add(documents=['我是王小鱼','我们一起学习RAG','今晚学习chromadb'],
                metadatas=[{'chapter':"3",'verse':'16'},{'chapter':"2",'verse':'10'},{'chapter':"5",'verse':'8'}],
                ids=['id1','id2','id3'],
                )
    embeddings = np.random.rand(3, 384).tolist()  # 假设每个文档的嵌入向量是768维的随机数
    collection.add(documents=['我是王小鱼','我们一起学习RAG','今晚学习chromadb'],
                metadatas=[{'chapter':"3",'verse':'16'},{'chapter':"2",'verse':'10'},{'chapter':"5",'verse':'8'}],
                embeddings=embeddings,
                ids=['id4','id5','id6'],
                )


    #查看数据
    print(collection.peek())
