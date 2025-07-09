import requests
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.environ["QWEATHER_API_KEY"]

client = OpenAI(
    api_key=os.environ["ZHIPU_API_KEY"],
    base_url=os.environ["ZHIPU_API_BASE"]
)


def get_weather_forecast(location_id):
    """
    从和风天气API获取未来24小时的天气预报。
    """
    # API URL 和地点参数
    base_url = "https://pd3yfqcqgq.re.qweatherapi.com/v7/weather/24h"
    
    # 将参数打包成字典
    params = {
        'location': location_id,
        # 注意：在实际应用中，请勿将密钥硬编码到代码中。
        'key': API_KEY
    }
    
    try:
        # 发送 GET 请求, requests 会自动将 params 编码到 URL 中
        # requests 也会自动处理 gzip 解压
        response = requests.get(base_url, params=params)
        
        # 检查请求是否成功 (状态码 200)
        response.raise_for_status()
        
        # 和风天气的API返回的是JSON格式数据
        weather_data = response.json()
        
        return weather_data
        
    except requests.exceptions.RequestException as e:
        print(f"请求过程中发生错误: {e}")
    except json.JSONDecodeError:
        print("无法解析返回的JSON数据。")
        print("收到的响应内容:", response.text)

def get_weather_location(location):
    """
    查询地理位置信息
    """
    url = "https://pd3yfqcqgq.re.qweatherapi.com/geo/v2/city/lookup"
    
    params = {
        "location": location,
        "key": API_KEY  # 和风天气API使用'key'参数而不是Bearer token
    }
    
    headers = {
        "Accept-Encoding": "gzip, deflate"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_location",
            "description": "根据城市名称查询其唯一的地理位置ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市、区或县的名称，例如：昌平, 北京"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "根据地理位置ID获取未来24小时的天气预报",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_id": {
                        "type": "string",
                        "description": "地点的ID，必须通过 get_weather_location 函数获取"
                    }
                },
                "required": ["location_id"]
            }
        }
    }
]

def get_weather_response(user_query):
    messages = []
    messages.append({"role": "system", "content": ("你是一个专门用于天气查询的助手。你不会询问用户任何问题，你只会根据用户提供的信息直接调用工具进行查询。")})
    messages.append({"role": "user", "content": user_query})
    
    response = client.chat.completions.create(
        model=os.environ["MODEL"],
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    while response.choices[0].message.tool_calls != None:
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        
        function = globals()[tool_call.function.name]
        function_result = function(**args)

        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool",
            "content": f"{json.dumps(function_result)}",
            "tool_call_id": tool_call.id
        })

        response = client.chat.completions.create(
            model="glm-4-plus",
            messages=messages,
            tools=tools,
        )

    return response.choices[0].message.content
