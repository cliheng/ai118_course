from loguru import logger as log
import requests



WEATHER_SEARCH={
    "type":"function",
    "function":{
        "name":"get_current_weather",
        "description":"获取某个城市或地区的的天气预报信息",
        "parameters":{
            "type":"object",
            "properties":{
                "city":{
                    'type':"string",
                    "description":"城市名称"
                }
            },
            "required":["city"]
        }
    }
}


def get_current_weather(city):
    log.info(f"get_city_coordinates: city={city}")
    api_key = "5ce7bc55834975dc2dcb65d95d0e0cae"  
    # 基本参数配置
    apiurl = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=5&appid={api_key}"  # 接口请求URL
    response = requests.get(apiurl)
    data = response.json()

    if response.status_code == 200:
        lat = data[0]["lat"]
        lon = data[0]["lon"]
        log.info(f'查询 {city} 经度: {lat}, 纬度: {lon}')
    else:
        return '未查询到该城市'

    base_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    response = requests.get(base_url)

    if response.status_code == 200:
        data = response.json()
        log.info(f'查询 {city} 天气: {data}')
        weather = {
            'temperature': data['main']['temp'],
            'description': data['weather'][0]['description'],
            'city': data['name'],
            'country': data['sys']['country']
        }
        return weather
    else:
        return {"error": data.get("message", "An error occurred.")}