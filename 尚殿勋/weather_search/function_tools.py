import requests
import json

MULTIPLY_TWO_NUMBERS = {

    
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
            "description": "通过百度搜索获得信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查询关键字"
                    }
                },
                "required": [
                    "query"
                ]
            }
        }
}

MONEY_SEARCH = {
    "type": "function",
    "function": {
        "name": "money_search",
        "description": "查询人民币与外币的汇率",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

WEATHER_SEARCH = {
      "type": "function",
    "function": {
        "name": "weather_search",
        "description": "根据用户的输入，查询当地的天气怎么样",
        "parameters": {
            "type": "object",
            "properties": {
                "query":{
                    "type":"string",
                    "description":"要查询的天气的城市名"
                }

            },
            "required": ["query"]
        }
    }
}

def multiply_two_numbers(a,b):
    result = a * b
    return result

def add_two_numbers(a,b):
    result = a + b
    return result

def baidu_search(query):


    uri = "https://qianfan.baidubce.com/v2/ai_search"
    heads = {
        "Authorization":"Bearer bce-v3/ALTAK-bIDfyfGjQUvs9U3SS0gHf/200f9c2acae512cdeb3c9335647d947f47cbd7bd",
        "Content-type":"application/json"
    }


    response = requests.post(
        uri,
        json=
            {
            "messages":[
                {
                    "role":"user",
                    "content": query
                }
            ]
                

                },

            headers=heads    
        
    )

    result = json.loads(response.text)
    return f'{result["references"]}'

def money_search():
    
    # 40-人民币牌价 - 代码参考（根据实际业务情况修改）

    # 基本参数配置
    apiUrl = 'http://web.juhe.cn/finance/exchange/rmbquot'
    apiKey = '0a8dd4de0770089136eb52882dca413b'
    requestParams = {
            'key': apiKey,
            'type': '',
            'bank': '',
        }

    # 发起接口网络请求
    response = requests.get(apiUrl, params=requestParams)

    # 解析响应结果
    if response.status_code == 200:
        responseResult = response.json()
        # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
        # print(responseResult)
        return responseResult['result']
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        print('请求异常')
    # 发起接口网络请求
    # response = requests.get(apiUrl, params=requestParams)
   
def weather_search(query):

    # 1213-根据城市查询天气 - 代码参考（根据实际业务情况修改）

    # 基本参数配置
    apiUrl = 'http://apis.juhe.cn/simpleWeather/query'  # 接口请求URL
    apiKey = '8872e582b5d8d66e931d34259a8527ce'  # 在个人中心->我的数据,接口名称上方查看

    
    # 接口请求入参配置
    requestParams = {
        'key': apiKey,
        'city': query,
    }

    # 发起接口网络请求
    response = requests.get(apiUrl, params=requestParams)

    # 解析响应结果
    if response.status_code == 200:
        responseResult = response.json()
        # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
        # print(responseResult)
        return responseResult['result']
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        print('请求异常')

# if __name__ == "__main__":
#     weather_search('北京')