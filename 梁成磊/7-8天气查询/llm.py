from openai import OpenAI
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
import os
import json
import function_tools

load_dotenv(find_dotenv())


# 调用openai模型
def openai_llm(model, temperature, max_tokens, question):
    # 创建openai客户端
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"]
    )

    # 创建信息列表
    messages = [{"role": "user", "content": question}]

    # 定义可用的函数工具（function calling）
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather_search",
                "description": "查询某一地区的天气预报",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "要查询天气情况的城市名称",
                        },
                    },
                    "required": ["city"],  # 说明两个参数是必须的
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cityair_search",
                "description": "查询某一地区的空气质量情况",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "要查询空气质量状况的城市名称",
                        },
                    },
                    "required": ["city"],  # 说明两个参数是必须的
                },
            },
        },
    ]

    # 调用模型进行对话，获取结果
    response = client.chat.completions.create(
        model=model,
        tools=tools,  # type: ignore
        messages=messages,  # type: ignore
        temperature=temperature,
        max_tokens=max_tokens,
        tool_choice="auto",
    )

    # 如果模型回复中包含函数调用请求，则循环处理
    if response.choices[0].message.tool_calls is not None:
        # 记录本次模型回复到消息列表
        messages.append(response.choices[0].message)  # type: ignore

        # 获取函数调用的参数（字符串形式），并解析为字典
        args = response.choices[0].message.tool_calls[0].function.arguments
        args = json.loads(args)

        # 获取要调用的函数名
        function_name = response.choices[0].message.tool_calls[0].function.name

        # 从function_tools模块中获取对应的函数
        invoke_run = getattr(function_tools, function_name)

        # 调用函数并获取结果
        result = invoke_run(**args)

        # 将函数调用结果以tool角色的消息形式加入消息列表
        messages.append(
            {
                "role": "tool",
                "content": f"{json.dumps(result)}",
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
            }
        )

        # 再次调用OpenAI接口，继续对话流程，直到没有新的函数调用
        res = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return res.choices[0].message.content


# 调用ZhipuAI模型
def zhipu_llm(model, temperature, max_tokens, question):
    # 创建openai客户端
    client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])

    # 创建信息列表
    messages = [{"role": "user", "content": question}]

    # 定义可用的函数工具（function calling）
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather_search",
                "description": "查询某一地区的天气预报",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "要查询天气情况的城市名称",
                        },
                    },
                    "required": ["city"],  # 说明两个参数是必须的
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cityair_search",
                "description": "查询某一地区的空气质量情况",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "要查询空气质量状况的城市名称",
                        },
                    },
                    "required": ["city"],  # 说明两个参数是必须的
                },
            },
        },
    ]

    # 调用模型进行对话，获取结果
    response = client.chat.completions.create(
        model=model,
        tools=tools,  # type: ignore
        messages=messages,  # type: ignore
        temperature=temperature,
        max_tokens=max_tokens,
        tool_choice="auto",
    )

    # 如果模型回复中包含函数调用请求，则循环处理
    if response.choices[0].message.tool_calls is not None:  # type: ignore
        # 记录本次模型回复到消息列表
        messages.append(
            {
                "role": response.choices[0].message.role,  # type: ignore
                "content": response.choices[0].message.content,  # type: ignore
            }
        )

        # 获取函数调用的参数（字符串形式），并解析为字典
        args = response.choices[0].message.tool_calls[0].function.arguments  # type: ignore
        args = json.loads(args)

        # 获取要调用的函数名
        function_name = response.choices[0].message.tool_calls[0].function.name  # type: ignore

        # 从function_tools模块中获取对应的函数
        invoke_run = getattr(function_tools, function_name)

        # 调用函数并获取结果
        result = invoke_run(**args)

        # 将函数调用结果以tool角色的消息形式加入消息列表
        messages.append(
            {
                "role": "tool",
                "content": f"{json.dumps(result)}",
                "tool_call_id": response.choices[0].message.tool_calls[0].id,  # type: ignore
            }
        )

        # 再次调用OpenAI接口，继续对话流程，直到没有新的函数调用
        res = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return res.choices[0].message.content  # type: ignore


# 调用百炼模型
def bailian_llm(model, temperature, max_tokens, question):
    # 创建openai客户端
    client = OpenAI(
        api_key=os.environ["ALIYUN_API_KEY"], base_url=os.environ["ALIYUN_BASE_URL"]
    )

    # 创建信息列表
    messages = [{"role": "user", "content": question}]

    # 定义可用的函数工具（function calling）
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather_search",
                "description": "查询某一地区的天气预报",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "要查询天气情况的城市名称",
                        },
                    },
                    "required": ["city"],  # 说明两个参数是必须的
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cityair_search",
                "description": "查询某一地区的空气质量情况",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "要查询空气质量状况的城市名称",
                        },
                    },
                    "required": ["city"],  # 说明两个参数是必须的
                },
            },
        },
    ]

    # 调用模型进行对话，获取结果
    response = client.chat.completions.create(
        model=model,
        tools=tools,  # type: ignore
        messages=messages,  # type: ignore
        temperature=temperature,
        max_tokens=max_tokens,
        tool_choice="auto",
    )

    # 如果模型回复中包含函数调用请求，则循环处理
    if response.choices[0].message.tool_calls is not None:
        # 记录本次模型回复到消息列表
        messages.append(response.choices[0].message)  # type: ignore

        # 获取函数调用的参数（字符串形式），并解析为字典
        args = response.choices[0].message.tool_calls[0].function.arguments
        args = json.loads(args)

        # 获取要调用的函数名
        function_name = response.choices[0].message.tool_calls[0].function.name

        # 从function_tools模块中获取对应的函数
        invoke_run = getattr(function_tools, function_name)

        # 调用函数并获取结果
        result = invoke_run(**args)

        # 将函数调用结果以tool角色的消息形式加入消息列表
        messages.append(
            {
                "role": "tool",
                "content": f"{json.dumps(result)}",
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
            }
        )

        # 再次调用OpenAI接口，继续对话流程，直到没有新的函数调用
        res = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return res.choices[0].message.content
