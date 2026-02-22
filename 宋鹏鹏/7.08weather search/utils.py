import function_tools
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

def weather_search(city, model, temperature, query):
    if model == "gpt-3.5-turbo":
        client = OpenAI(
            api_key=os.environ["openai_api_key"],
            base_url=os.environ["openai_api_base"],
        )
    elif model == "glm-4-flash":
        client = OpenAI(
            api_key=os.environ["zhipu_api_key"],
            base_url=os.environ["zhipu_api_base"],
            
        )
    elif model == "qwen1.5-0.5b-chat":
        client = OpenAI(
            api_key=os.environ["bai_api_key"],
            base_url=os.environ["bai_api_base"],
            
        )
    tools = [function_tools.WEATHER_SEARCH]
    messages = [
        {'role': 'system', "content": "不需要要求用户补充问题，直接按问题调用tool"},
        {"role": "user", "content": query}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature
    )
    while response.choices[0].message.tool_calls is not None:
        messages.append(response.choices[0].message)
        for tool_call in response.choices[0].message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            function_name = tool_call.function.name
            invoke_fun = getattr(function_tools, function_name)
            result = invoke_fun(**args)
            messages.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools
        )
    result = (response.choices[0].message.content)
    return result
