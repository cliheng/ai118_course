import requests
# import json

import logging

logging.basicConfig(level=logging.INFO)

def get_current_weather(city):
    logging.info(f"查询{city}天气开始")
    api_key = "592e693e63cdc745ccf479256bbf341e"

    apiurl = f'https://api.openweathermap.org/geo/1.0/direct?q={city}&limit=5&appid={api_key}'
    response = requests.get(apiurl)
    data = response.json()

    if response.status_code == 200 and data:
        lat = data[0]["lat"]
        lon = data[0]["lon"]
        logging.info(f'获取{city}坐标: 纬度{lat}, 经度{lon}')
    else:
        return '未查询到该城市'
    
    base_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=zh_cn"
    response = requests.get(base_url)
    data = response.json()

    if response.status_code == 200:
        weather_info = {
            '城市': city,
            '温度(℃)': data['main']['temp'],
            '体感温度(℃)': data['main']['feels_like'],
            '天气状况': data['weather'][0]['description'],
            '湿度(%)': data['main']['humidity'],
            '风速(m/s)': data['wind']['speed']
        }
        logging.info(f"天气查询结果: {weather_info}")
        return weather_info
    else:
        logging.error(f"天气查询失败: {data.get('message','未知错误')}")
        return {"error": "天气查询失败"}

WEATHER_SEARCH = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "获取某个城市或地区的天气预报信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }
    }
}