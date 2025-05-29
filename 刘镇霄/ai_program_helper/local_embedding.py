from time import time
from typing import List

from openai import OpenAI

def __init__(self, api_key: str):
    self.client = OpenAI(api_key=api_key, base_url="http://localhost:11434/v1/")

def get_embedding(self, text: str) -> List[float]:
    try:
        response = self.client.embeddings.create(
            model="bge-m3",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"OpenAI API 错误: {e}")
        return None

def local_embeddings(self, texts: List[str]):
    embeddings = []
    valid_texts = []
    for i, text in enumerate(texts):
        print(f"\n处理进度: {i + 1}/{len(texts)} - 测试文本: {text[:20]}...")

        start_time = time()
        embedding = self.get_embedding(text)
        process_time = time() - start_time

        print(f"处理时间: {process_time:.2f}秒")
        if embedding:
            print(f"嵌入维度: {len(embedding)}")
            embeddings.append(embedding)
            valid_texts.append(text)

    if embeddings:
        self.visualize_embeddings(embeddings, valid_texts)