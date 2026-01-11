import torch
import dashscope
from http import HTTPStatus
 
def embed_with_qwen(text_list):
    dashscope.api_key = 'sk-6b863ea131814d2da0148c9dd72be277'
    embedding_list = []
    # print("断点1")
    sum=0
    for text in text_list:
        sum=sum+1
        print(f"正在向量化中{sum}/{len(text_list)}...")
        resp = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v2,
            input=text)
        if resp.status_code == HTTPStatus.OK:
            embedding = resp.output['embeddings'][0]['embedding']
            embedding_list.append(embedding)
        else:
            print(resp)
            return resp
    # print("断点2")
    return embedding_list
 

def embed_with_bge(text_list, tokenizer, model):
 
    # 对于短查询到长文档的检索任务, 为查询加上指令
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

    encoded_input = tokenizer(text_list, max_length=512, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
 
        # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings_list = sentence_embeddings.tolist()
    return sentence_embeddings_list