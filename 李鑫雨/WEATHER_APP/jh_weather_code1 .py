from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import function_weather

if __name__ == '__main__':
    load_dotenv()
    client = OpenAI(
        base_url=os.environ['BASE_URL'],
        api_key=os.environ['API_KEY']
    )

tools=[
    function_weather.HUIJU_SEARCH
]

messages=[{"role": "system", "content": "北京天气"}]
messages=[{"role": "user", "content": "调用{tools}帮我查询的天气"}]
response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
        )

while response.choices[0].message.tool_calls is not None:
    messages.append(response.choices[0].message)

    for tool_call in response.choices[0].message.tool_calls:
        args=tool_call.function.arguments
        args=json.loads(args)
    
        function_name =tool_call.function.name

        invoke_fun =getattr(function_weather,function_name)
        result=invoke_fun(**args)

        messages.append(
            {
                "role": "tool",
                "content":f"{json.dumps(result)}",
                "tool_call_id":tool_call.id
            }

        )

        response=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )

print(response.choices[0].message.content)