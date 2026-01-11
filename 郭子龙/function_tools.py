import logging
import requests
import json


MULTIFY_TWO_NUMBERS ={
        "type": "function",
        "function": {
            "name": "multiply_two_numbers",
            "description": "两个数相乘",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "第一个数字"
                    },
                    "b":{
                        "type": "number",
                        "description": "第二个数字"
                    }
                },
                "required": [
                    "a", "b"
                ]
            }
        }
    }
ADD_TWO_NUMBERS = {
        "type": "function",
        "function": {
            "name": "add_two_numbers",
            "description": "两个数相加",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "第一个数字"
                    },
                    "b":{
                        "type": "number",
                        "description": "第二个数字"
                    }
                },
                "required": [
                    "a", "b"
                ]
            }
        }
    }
BAIDU_SEARCH = {
        "type": "function",
        "function": {
            "name": "baidu_search",
            "description": "根据用户提供的信息通过互联网搜索引擎查询对应问题的信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查询关键词"
                        }
                    },
                "required":["query"]
            }
        }
    }

HUILV_SEARCH = {
        "type": "function",
        "function": {
            "name": "huilv_search",
            "description": "查询人民币与外币的汇率信息，单位为100外币",
            "parameters": {
                "type": "object",
                "properties": {},
                "required":[]
            }
        }
    }

DIVIDE_TWO_NUMBERS = {
        "type": "function",
        "function": {
            "name": "divide_two_numbers",
            "description": "两个数相除",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "被除数"
                    },
                    "b":{
                        "type": "number",
                        "description": "除数"
                    }
                },
                "required": [
                    "a", "b"
                ]
            }
        }
    }

WEATHER_SEARCH = {
        "type": "function",
        "function": {
            "name": "weather_search",
            "description": "获取指定城市的天气数据",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }



def weather_search():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info("weather_search")
    # 基本参数配置
    apiUrl = 'http://apis.juhe.cn/simpleWeather/query'  # 接口请求URL
    apiKey = '18108b0f28a5a63960cd38108aab778f'  # 在个人中心->我的数据,接口名称上方查看

    # 接口请求入参配置
    requestParams = {
        'key':apiKey,
        'city': '苏州',
    }

    # 发起接口网络请求
    response = requests.get(apiUrl, params=requestParams)

    # 解析响应结果
    if response.status_code == 200:
        responseResult = response.json()
        return f'{responseResult["result"]}'
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        print('请求异常')
def divide_two_numbers(a, b):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info(f"divide_two_numbers:a={a},b={b}")
    result = a / b
    return result

def huilv_search ():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info("huilv_search")
    # 基本参数配置
    apiUrl = 'http://web.juhe.cn/finance/exchange/rmbquot'  # 接口请求URL
    apiKey = 'f3de2fad815094c91c3897b15445ea2c'  # 在个人中心->我的数据,接口名称上方查看

    # 接口请求入参配置
    requestParams = {
        'key': apiKey,
        'type': '',
        'bank': '',
    }

    # 发起接口网络请求
    response = requests.get(apiUrl, params=requestParams)
    # print("函数调用结果：",response.text)
    # 解析响应结果
    if response.status_code == 200:
        responseResult = response.json()
        return f'{responseResult["result"]}'
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        print('请求异常')
def multiply_two_numbers(a, b):
    result = a * b
    return result
def add_two_numbers(a, b):
    result = a + b
    return result

def baidu_search(query):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info(f"baidu_search:query={query}")
    uri = "https://qianfan.baidubce.com/v2/ai_search"
    #请求资源
    heads = {
        "Authorization":"Bearer bce-v3/ALTAK-BlRe1wiqh0cMauoYjeGWx/228e1e34c32712ecb394b7efc0fb30524be23dd4",
        "Content-Type":"application/json"
    }
    response = requests.post(
        uri,
        json={
            "messages":[
                {
                    "role":"user",
                    "content":query
                }
            ]
        },
        headers=heads
        )
    result = json.loads(response.text)
    return f'{result["references"]}'