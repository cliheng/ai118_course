import chromadb
import numpy as np
if __name__ == "__main__":
    #创建一个chroma客户端
    client = chromadb.HttpClient(host='localhost', port=8000)  
#     # collection = client.create_collection(
#     # name="collection2",
#     # metadata={"hnsw:space": "cosine"} # l2 is the default
# )
    collection = client.get_collection(name="collection2")  # 获取已存在的collection

# 指定向量
    collection.add(documents=['我是王小鱼','我们一起学习RAG','今晚学习chromadb'],
                metadatas=[{'chapter':"3",'verse':'16'},{'chapter':"2",'verse':'10'},{'chapter':"5",'verse':'8'}],
                embeddings=[[-0.04291284,  0.12459736,  0.07283306],[0.00072299,  0.1379851 ,  0.05514374],[0.05965715,  0.06796112,  0.03112736]],
                ids=['id4','id5','id6'],
                )
    print(collection.peek())

    collection.update(documents=['我是王小鱼','我们一起学习RAG','今晚学习chromadb'],
               metadatas=[{'chapter':"3",'verse':'16'},{'chapter':"2",'verse':'10'},{'chapter':"5",'verse':'8'}],
               embeddings=[[1.6,  1.2,  1.3],[0.00072299,  0.1379851 ,  0.05514374],[0.05965715,  0.06796112,  0.03112736]],
               ids=['id4','id5','id6'])



    # collection.upsert(documents=['我是王小鱼','我们一起学习RAG','今晚学习chromadb'],
    #            metadatas=[{'chapter':"3",'verse':'16'},{'chapter':"2",'verse':'10'},{'chapter':"5",'verse':'8'}],
    #            embeddings=[[1.6,  1.2,  1.3],[0.00072299,  0.1379851 ,  0.05514374],[0.05965715,  0.06796112,  0.03112736]],
    #            ids=['id7','id8','id9'])
print(collection.peek())