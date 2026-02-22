import requests

from backend.weather_tools.weather_api import weather_tool_api, weather_tool_base_url
# 工具函数
def weather_search(city):
    print(f"正在调用工具查询{city}的天气")
    requestParams = {
        'key': weather_tool_api(),
        'city': city,
    }
    response = requests.get(weather_tool_base_url(), params=requestParams)
    if response.status_code == 200:
        responseResult = response.json()
        print(responseResult['result'])
        return responseResult['result']
    else:
        print('请求异常')