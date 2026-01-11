#function调用天气预报软件 
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import f_tools


if __name__ == '__main__':

    load_dotenv()

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_API_BASE']
    )

    tools = [f_tools.WEATHER_SEARCH]

    messages = [
        {  # 第一个系统消息
            "role": "system",
            "content": "不需要要求用户补充问题，直接按问题调用tool"
        },
        {  # 第二个用户消息
            "role": "user",
            "content": "帮我查询一下shanxi的天气情况"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    while response.choices[0].message.tool_calls is not None:

        #记录函数调用
        messages.append(response.choices[0].message)

        for tool_call in response.choices[0].message.tool_calls:
        
            #调用参数
            args =tool_call.function.arguments
            args = json.loads(args)
            #函数名
            function_name = tool_call.function.name
            #调用函数
            invoke_fun = getattr(f_tools, function_name)#外部模块动态获取函数
            
            result = invoke_fun(**args)

            #结果添加messages，告知llm调用结果
            messages.append(
                {
                    "role": "tool",
                    "content": f"{json.dumps(result)}",
                    "tool_call_id": tool_call.id,
                }
            )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=tools
        )

    print(response.choices[0].message.content)