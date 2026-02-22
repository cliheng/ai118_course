import json
import os
import zhipuai
import chromadb
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from modelscope import AutoTokenizer, AutoModel
import torch

def parse_data(file_path):
    keywords,contents = [],[]
 
    with open(file_path,'r',encoding='utf-8') as f:
        data_source = json.load(f)

        for item in data_source:
            text = item['k_qa_content']
            key,content = text.split("#\n")
            keywords.append(key)
            contents.append({"content":content})
        return keywords,contents

def api_embedding(input_text,client,model_name):
        completion = client.embeddings.create(
            model=model_name,
            input=input_text,
            dimensions=1024,
            )
    

        embedding = completion.data[0].embedding
        return embedding

def load_embeddings(sentences):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    model.eval()

    # 编码输入
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # 获取嵌入向量
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    
    # normalize embeddings
    sentence_embeddings = sentence_embeddings.numpy().tolist()
    
    return sentence_embeddings


if __name__ == "__main__":
    
    load_dotenv(find_dotenv())  
    client = chromadb.HttpClient(host="localhost", port=8000)
    if not os.path.exists('chroma'):
         
        keywords,contents = parse_data('data_source.json')
        embeddings = load_embeddings(keywords)
        
        collection = client.get_or_create_collection("collection119")

    ids = []
    openai_client = OpenAI(  # 用不同的变量名
            api_key=os.environ['api_key'],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        ) 
    for i in range(len(contents)):
        ids.append(f"id{i}")

    collection.add(
         ids=ids,
         embeddings=embeddings,
         documents=keywords,
         metadatas=contents,
    )
    



    #print("完成")

    question = input("请输入问题：")
    #向量化
    q_emb = load_embeddings([question])
    collection = client.get_collection("collection119")  # 这里的client还是chromadb的
    result = collection.query(query_embeddings=q_emb,n_results=1)
    #提取结果中的metadata
    print("当前库中数据量：", collection.count())
    print("检索到的结果：", result)

    contents = ""
    if len(result['metadatas']) > 0 and len(result['metadatas'][0]) > 0:
        contents = result['metadatas'][0][0]['content']
    else:
        print("未检索到相关内容，可能是向量不匹配或数据未入库。")

    prompt = f"你是一个精通python语言编程的专家，回答用户提出的问题。回答内容，参考补充资料。\n\n补充资料：{contents}"
    print(prompt)