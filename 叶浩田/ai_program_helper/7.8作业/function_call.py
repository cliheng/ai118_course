from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import function_tools
from zhipuai import ZhipuAI  # 导入 ZhipuAI 类

city=input("请输入要查询的城市（如：北京）：")

if __name__ == "__main__":
    load_dotenv()

    # client = OpenAI(
    #     api_key=os.environ["ZHIPU_API_KEY"],
    #     base_url=os.environ["ZHIPU_API_BASE"]
    # )
    
    client = ZhipuAI(
        api_key=os.environ["ZHIPU_API_KEY"],
        base_url=os.environ["ZHIPU_API_BASE"]
    )
    

    tools= [function_tools.TIANQI_SEARCH]

    messages = [
        {'role': 'system', 'content':"不需要要求用户补充问题，直接按问题调用tool"},
        {"role": "user", "content": f"{city}的天气"}
    ]

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    while response.choices[0].message.tool_calls is not None:
        # 记录函数调用
        # messages.append(response.choices[0].message)
        message_dict = response.choices[0].message.model_dump()
        messages.append(message_dict)
        
        for tool_call in response.choices[0].message.tool_calls:
            # 调用参数
            args = tool_call.function.arguments
            args = json.loads(args)
            # 函数名
            fuction_name = tool_call.function.name
            # 调用函数
            # 外部模块动态获取函数
            invoke_fun = getattr(function_tools, fuction_name)
            result = invoke_fun(**args)

            # 结果添加messages，告知LLM调用结果
            messages.append(
                {
                    "role": "tool",
                    "content": f"{json.dumps(result)}",
                    "tool_call_id": tool_call.id
                }
            )

        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            tools=tools
        )

    print(response.choices[0].message.content)
