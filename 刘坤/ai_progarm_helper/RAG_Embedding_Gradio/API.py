import requests

#1.用于回复用户
url = "https://api.siliconflow.cn/v1/chat/completions"
#1.1用于图像识别
model = "Pro/Qwen/Qwen2.5-VL-7B-Instruct"
#1.2用于文本对话
model_text = "Qwen/Qwen3-8B"
payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": "What opportunities and challenges will the Chinese large model industry face in 2025?"
        }
    ],
}
headers = {
    "Authorization": "Bearer <your_key>",
    "Content-Type": "application/json"
}
response = requests.request("POST", url, json=payload, headers=headers)

#2.用于embedding计算
url = "https://api.siliconflow.cn/v1/embeddings"
payload = {
    "model": "BAAI/bge-large-zh-v1.5",
    "input": text,
    "encoding_format": "float"
}
headers = {
    "Authorization":"Bearer <your_key>",
    "Content-Type": "application/json"
}

