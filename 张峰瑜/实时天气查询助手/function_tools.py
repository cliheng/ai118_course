import requests
import json
import logging

WEATHER_SEARCH = [
    {
        "type": "function",
        "function": {
            "name": "weather_search",
            "description": "根据用户提供的城市信息通过互联网实时查询当地实时天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": [
                    "city"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather_forecast",
            "description": "根据用户提供的城市信息通过互联网查询该地未来天气预报",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": [
                    "city"
                ]
            }
        }
    }
]


def weather_search(city):
    log = logging.getLogger(__name__)
    log.info(f"weather_search：city={city}")
    # 请求资源
    apiKey = '8dbf0c36529c8aa1ae21639cda49ad49'
    apiUrl = 'http://apis.juhe.cn/simpleWeather/query'

    params = {
        'key': apiKey,
        'city': city
    }
    response = requests.get(apiUrl, params=params)
    data = response.json()


    if response.status_code == 200:
        data = response.json()
        log.info(f'查询 {city} 天气：{data}')
        result = data["result"]
        realtime = result["realtime"]
        weather = {
            'temperature': realtime["temperature"],
            'description': realtime["info"],
            'humidity': realtime["humidity"],
            'city': result["city"]
        }
        # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
        return weather
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        return {"error": data.get("message","An error occured.")}