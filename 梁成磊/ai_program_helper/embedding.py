"""
此模块用于调用向量模型将数据转化为向量
"""

from openai import OpenAI
from modelscope import AutoTokenizer, AutoModel
import torch
import os


# 闭源embedding模型API调用
def cloud_embedding(input_text, model_name):
    """
    调用阿里百炼平台向量模型进行数据向量化

    Args:
        input_text(list): 处理后的数据列表
        model_name(str): 调用模型名称

    Returns:
        list: 向量化处理后的列表
    """
    # 创建一个用于存储响亮的列表
    embedding_list = []

    # 创建openai客户端
    client = OpenAI(
        api_key=os.environ["api_key"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 遍历需要向量化的数据列表，逐条进行向量化
    for content in input_text:
        completion = client.embeddings.create(
            model=model_name, input=content, dimensions=768, encoding_format="float"
        )

        # 将返回的向量存入列表
        embedding_list.append(completion.data[0].embedding)

    # 返回向量化后的数据列表
    return embedding_list


# 开源embedding模型调用
def local_embedding(text_list):
    """
    调用modelscope中的BAAI/bge-large-zh-v1.5对数据进行向量化处理

    Args:
        text_list(list): 处理后的数据列表

    Returns:
        list: 向量化后的数据列表
    """
    # 自动分词器，将数据分成多个词元（token）
    # from_pretrained -->  用于加载预训练的模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")

    # embedding模型，用于生成数据的向量表示
    model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5")
    model.eval()

    # 对输入文本进行处理，返回张量格式的结果
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )

    # 生成embedding
    # 使用模型生成文本嵌入向量
    # torch.no_grad()避免计算梯度，节省内存并加速计算
    with torch.no_grad():
        # 将编码后的输入传入模型，获取输出
        # **encoded_input 解包字典，包含 input_ids、attention_mask 等输入
        model_output = model(**encoded_input)

        # 提取[CLS]标记对应的隐藏状态作为整个句子的表示
        # model_output[0]是最后一层的隐藏状态，形状为 [batch_size, sequence_length, hidden_size]
        # [:, 0]选择每个句子的第一个位置（即[CLS]标记）的向量表示
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = sentence_embeddings.numpy().tolist()

    return sentence_embeddings
