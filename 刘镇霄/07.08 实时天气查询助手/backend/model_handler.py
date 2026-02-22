import json
from backend import tool_manager
from openai import OpenAI
from backend.tool_manager import WEATHER_SEARCH
from config.config import bigmodel_api_key, bigmodel_base_url, bailian_api_key, bailian_base_url, api2d_api_key, \
    api2d_base_url
from config.prompts import example_prompt
from utils.logger import logger

# 后端大模型调用
def main(question, model, temperature, max_tokens, platform):
    api_key = None
    base_url = None
    # 前端不同平台对应不同api、url
    if platform == "智谱":
        api_key = bigmodel_api_key()
        base_url = bigmodel_base_url()
    elif platform == "阿里云百炼":
        api_key = bailian_api_key()
        base_url = bailian_base_url()
    elif platform == "API2D":
        api_key = api2d_api_key()
        base_url = api2d_base_url()
    # 创建OpenAI客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    messages = [
        {"role": "system",
         "content": f'你是一个 严格遵循流程 的天气查询AI，必须 使用tools获取数据，禁止自行编造信息,必须查看提示词模板{example_prompt()}'},
        {"role": "user", "content": question}
    ]
    tools = [WEATHER_SEARCH]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    while response.choices[0].message.tool_calls is not None:
        messages.append(response.choices[0].message)  # 添加AI的消息
        tool_responses = []
        for tool_call in response.choices[0].message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            fuction_name = tool_call.function.name
            invoke_fun = getattr(tool_manager, fuction_name)
            result = invoke_fun(**args)
            tool_responses.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fuction_name,
                "content": str(result)
            })
        messages.extend(tool_responses)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # 生成日志
        logger.info(response.choices[0].message.content)
        return response.choices[0].message.content
