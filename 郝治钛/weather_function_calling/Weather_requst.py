import requests
import logging as log
import os 
from dotenv import load_dotenv

load_dotenv()
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# def weather(query):
#     log.info(f"weather: q={query}")
#    # 1213-根据城市查询天气 - 代码参考（根据实际业务情况修改）

#     # 基本参数配置
#     apiUrl = 'http://apis.juhe.cn/simpleWeather/query'  # 接口请求URL
#     apiKey = '8872e582b5d8d66e931d34259a8527ce'  # 在个人中心->我的数据,接口名称上方查看

#     # 接口请求入参配置
#     requestParams = {
#         'key': apiKey,
#         'city': query,
#     }

#     # 发起接口网络请求
#     response = requests.get(apiUrl, params=requestParams)

#     # 解析响应结果
#     if response.status_code == 200:
#         responseResult = response.json()
#         # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
#         # print(responseResult['result'])
#         return responseResult['result']
#     else:
#         # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
#         # print('请求异常')
#         return '请求异常'
    
# WEATHER = {
#     'type': 'function',
#     'function': {
#         'name': 'weather',
#         'description': '查询天气',
#         'parameters': {
#             'type' : 'object',
#             'properties' : {
#                 'query' : {
#                     'type' : 'string',
#                     'description' : '查询当地天气关键词'
#                 }},
#             'required' : ['query'] 
#         }
#     }
# }

def openweather(city):
    log.info(f"weather: city={city}")
    response = requests.get(f'http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=5&appid={os.environ['WEATHER_API_KEY']}')

    data = response.json()
    # 解析响应结果
    if response.status_code == 200:
        lat = data[0]['lat']
        lon = data[0]['lon']
        
        
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        print('地址请求异常')
        return '未查询到该城市'
    response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={os.environ['WEATHER_API_KEY']}")
    data = response.json()
    if response.status_code == 200:
        # print(data)
        return data
    else:
        print('经纬度错误')
        return '查询失败'
OPEN_WEATHER = {
    'type': 'function',
    'function': {
        'name': 'openweather',
        'description': '查询天气',
        'parameters': {
            'type' : 'object',
            'properties' : {
                'city' : {
                    'type' : 'string',
                    'description' : '查询当地天气关键词'
                }},
            'required' : ['city'] 
        }
    }
}


















# if __name__ == '__main__':
#     openweather('苏州')