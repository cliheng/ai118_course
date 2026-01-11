from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import function_tools

if __name__ == "__main__":

    load_dotenv()

    client = OpenAI(
        api_key=os.environ["ZHIPU_API_KEY"],
        base_url=os.environ["ZHIPU_API_BASE"]
    )
    tools = function_tools.WEATHER_SEARCH

    messages = [
        {"role": "system", "content": "不需要让用户补充问题，直接根据用户输入调用合适的tool。"},
        {"role": "user", "content": "北京今天天气怎么样？"}
    ]

    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
while response.choices[0].message.tool_calls != None:
    # 返回工具调用信息
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print("工具调用信息：", args)

    # 调用其他模块中的函数
    function = getattr(function_tools, tool_call.function.name)
    function_result = function(**args)
    print("工具调用结果：", function_result)

    # 将函数调用和调用结果添加到消息列表中
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "content": f"{json.dumps(function_result)}",
        "tool_call_id": tool_call.id
    })

    response = client.chat.completions.create(
        model="glm-4-plus",  # 填写需要调用的模型名称
        messages=messages,
        tools=tools,
    )

result = response.choices[0].message.content
print("大模型输出信息", result)