from openai import OpenAI
import re

def qwen3_ai_q(query_keyword,content):
    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1/")
    print("助手正在奋笔疾书...")
    response = client.chat.completions.create(
        model="qwen3:1.7b",
        messages=[
            {"role": "system", "content": "你是一个专业风趣的大陆python老师，致力于详细的回答"},
            {"role": "user", "content": f"深呼吸，我想了解有关{query_keyword}的问题，请结合材料做出回复，材料是：{content}"},
        ],
        stream=False
    )
    return re.sub(r'<think>.*?</think>', '', response.choices[0].message.content, flags=re.DOTALL)