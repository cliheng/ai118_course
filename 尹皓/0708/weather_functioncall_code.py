import logging
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import functioncall_tools

# 设置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 文件处理器
file_handler = logging.FileHandler('weather_app.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

if __name__ == "__main__":
    load_dotenv()

    client = OpenAI(
        api_key = os.environ["OPENAI_API_KEY"],
        base_url = os.environ["OPENAI_BASE_URL"]
    )

    tools = [
        functioncall_tools.WEATHER_SEARCH
    ]

    messages = [
        {"role": "system", "content": "不需要要求用户补充问题,直接按问题调用tool"},
        {"role": "user", "content": "北京今天天气如何?"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    while response.choices[0].message.tool_calls is not None:

        messages.append(response.choices[0].message)
        
        for tool_call in response.choices[0].message.tool_calls:
            if tool_call.type == "function":
                args = tool_call.function.arguments
                args = json.loads(args)
                function_name = tool_call.function.name
                invoke_fun = getattr(functioncall_tools, function_name)
                result = invoke_fun(**args)
                
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
                tools=tools,
            )
    print(response.choices[0].message.content)