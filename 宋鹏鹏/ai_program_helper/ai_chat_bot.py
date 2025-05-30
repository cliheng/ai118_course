#导入大模型基于api调用，实现智能聊天机器人
#接收用户问题，对用户问题进行向量化，跟数据库中的向量进行对比，找到最相似的向量，返回对应的答案content
#根据用户问题，结合数据库中的资料，生成回答

import os
from openai import OpenAI
from zhipuai import ZhipuAI

# 调用向量化api，接受用户问题，返回向量
def question_embedding(question, client):
    """
    接受用户问题，返回向量
    input: question - 用户输入的问题
           client - OpenAI客户端
    output: q_embs - 问题的向量表示列表，只有一个元素
    """
    # 实现用户输入问题的向量化
    output = client.embeddings.create(
        model="text-embedding-v3",
        input=[question],  # 需要列表
        dimensions=768,
        encoding_format="float"
    )
    q_emb = output.data[0].embedding
    return q_emb


#调用智谱大模型实现问题回答，目前有两个参数，用户问题question，数据库的资料date

prompt_template = (
    "作为Python编程专家(精通Python语言设计、代码分析与错误诊断)"
    "请根据以下输入要素处理问题："
    "1) 用户问题：{message}（需解决的编程问题或需求）；"
    "2) 参考资料：{content}（相关技术文档或上下文）。"
    "输出要求：使用Markdown结构化排版，内容需包含："
    "a) 基于参考资料的解决方案；"
    "b) 对图片中代码/错误信息的解析；"
    "c) 可执行的优化建议。"
    "d) 著明对资料的引用"
    )

def chat_with_model(message, content):
    """
    与智谱大模型进行对话，获取回答
    input: message - 用户输入的问题, content - 数据库中的资料
    
    output: answer - 模型的回答
    """
    prompt = prompt_template.format(message=message, content=content)
    
    client = ZhipuAI(
        api_key=os.getenv("ZHIPUAI_API_KEY"),  # 从环境变量中获取API密钥
        base_url="https://open.bigmodel.cn/api/paas/v4"  # 智谱大模型的base_url
    )
    response = client.chat.completions.create(
        model="glm-4-flash-250414",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    answer = response.choices[0].message.content
    return answer