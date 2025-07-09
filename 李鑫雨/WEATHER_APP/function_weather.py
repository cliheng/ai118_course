import requests
import logging
import json 
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)


HUIJU_SEARCH={
        "type": "function",
        "function": {
            "name": "huiju_weather_search",
            "description": "根据城市查询天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称, e.g. 北京"
                    }
                },
                "required": ["city"]
            }
        }
}
# 1213-根据城市查询天气 - 代码参考（根据实际业务情况修改）
def huiju_weather_search(city):
    logging.info("根据城市查询天气={}".format(city))
    
    # 从环境变量获取 API Key
    api_key = os.environ.get('JUHE_API_KEY')
    if not api_key:
        return {'error': '未配置聚合数据API Key，请在.env文件中设置JUHE_API_KEY'}
    
    # 基本参数配置
    apiUrl = 'http://apis.juhe.cn/simpleWeather/query'  # 接口请求URL

    # 接口请求入参配置
    requestParams = {
        'key': api_key,
        'city': city,
    }

    try:
        # 发起接口网络请求
        response = requests.get(apiUrl, params=requestParams)
        
        # 解析响应结果
        if response.status_code == 200:
            responseResult = response.json()
            if responseResult.get('error_code') == 0:
                return responseResult
            else:
                return {'error': f"API返回错误：{responseResult.get('reason', '未知错误')}"}
        else:
            return {'error': f'请求失败，状态码：{response.status_code}'}
    except requests.exceptions.RequestException as e:
        return {'error': f'网络请求异常：{str(e)}'}
    except json.JSONDecodeError:
        return {'error': '解析响应数据失败'}
    


client_openai = OpenAI(
    base_url=os.environ['BASE_URL'],
    api_key=os.environ['API_KEY']
    )

client_zhipu=OpenAI(
    base_url=os.environ['ZHIPU_URL'],
    api_key=os.environ['ZHIPU_API']
)
client_qianwen = OpenAI(
    api_key=os.environ['QIANWEN_API'],
    base_url=os.environ['QIANWEN_URL'],
)

def op_weather_search(city):
    logging.info("op_ city_coordingates:city{city}")
