import chromadb
import numpy as np
if __name__ == "__main__":
    #创建一个chroma客户端
    client = chromadb.HttpClient(host='localhost', port=8000)  


    collection = client.get_collection(name="collection_name")  # 获取已存在的collection


    #查询向量
    query = np.random.rand(384) #生成一个随机向量，长度为384

    #查询
    query = collection.query(query_embeddings=query,
                             n_results=1  #返回 1个结果
                             )  

    
    
    
    #查看数据
    print(query)
