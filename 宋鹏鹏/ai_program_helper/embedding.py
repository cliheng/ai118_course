import json
#数据预处理


def process(filename):
    """
    处理json文件，提取关键词和内容
    input: filename - json文件名
    output: keywords - 关键词列表, concents - 内容列表
    """
    keywords = []
    concents = []
    with open(filename, 'r', encoding='utf-8') as f:
        #读取json文件
        data = json.load(f)
        for item in data:
            # 提取关键词和内容
            lst = item['k_qa_content']
            keyword, concent = lst.split('#\n', 1)
            keywords.append(keyword)
            concents.append(concent)
    return keywords, concents


def api_embed(keywords, client):
    """
    调用API生成嵌入
    input: keywords - 关键词列表
           client - OpenAI客户端
    output: embeddings - 嵌入列表
    """

    embeddings = []
    batch_size = 10  # 批处理大小
    for i in range(0, len(keywords), batch_size):
        input_text = keywords[i:i + batch_size]  # 分批处理关键词
        
        # 调用API生成嵌入
        output = client.embeddings.create(
            model="text-embedding-v3",
            input=input_text,  # 使用process函数返回的关键词列表
            dimensions=768,
            encoding_format="float"
        )
        # 提取每个embedding
        batch_embeddings = [item.embedding for item in output.data]
        embeddings.extend(batch_embeddings)
    
    return embeddings

