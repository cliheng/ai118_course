import requests
import json
import logging


TIANQI_SEARCH = {
    "type": "function",
    "function": {
        "name": "tianqi_search",
        "description": "根据用户提供的城市名称,查询该城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "需要查询天气的城市名称，如：北京、上海等"
                }
            },
            "required": ["city"]
        }
    }
}


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def tianqi_search(city):
    logging.info(f"tianqi_search: city={city}")
    # 1213-根据城市查询天气 - 代码参考（根据实际业务情况修改）

    # 基本参数配置
    apiUrl = 'http://apis.juhe.cn/simpleWeather/query'  # 接口请求URL
    apiKey = '2a9ce2bb111db1b4f39fa5439dc1c3a4'  # 在个人中心->我的数据,接口名称上方查看

    # 接口请求入参配置
    requestParams = {
        'key': apiKey,
        'city': city,
    }

    # 发起接口网络请求
    response = requests.get(apiUrl, params=requestParams)

    # 解析响应结果
    if response.status_code == 200:
        responseResult = response.json()
        # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
        return responseResult
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        print('请求异常')