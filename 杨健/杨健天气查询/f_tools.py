import requests
import logging as log
import requests

# 在文件顶部添加
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# 1213-根据城市查询天气 - 代码参考（根据实际业务情况修改）

# 修改工具定义结构（原代码第19行附近）
# 原工具定义可能嵌套了多余的层级，修改为：
GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "根据城市名称获取天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如：北京"
                }
            },
            "required": ["city"]
        }
    }
}

def get_weather(city):
    # log.info('get_weather')
    # 基本参数配置
    apiUrl = 'http://apis.juhe.cn/simpleWeather/query'  # 接口请求URL
    apiKey = 'bbbd613a3bb7940c336a29b5cca684be'  # 在个人中心->我的数据,接口名称上方查看

    # 接口请求入参配置
    requestParams = {
        'key': apiKey,
        'city': city,  #此处必须使用函数参数
    }

    # 发起接口网络请求
    response = requests.get(apiUrl, params=requestParams)

    # 解析响应结果
    if response.status_code == 200:
        responseResult = response.json()
        # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
        return f'{responseResult["result"]}'
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        print('请求异常')

def get_current_weather(city):
    logger.info(f"获取城市坐标: city={city}")
    api_key = 'a3783959f12ab6a99d18848b71e92787'  # 替换为你的API密钥
    
    # 修复1: 正确格式化地理编码URL
    geo_url = f'http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}'
    response = requests.get(geo_url)
    
    if response.status_code != 200:
        return {"error": f"地理编码请求失败: {response.status_code}"}
    
    data = response.json()
    if not data:
        return {"error": "未找到该城市"}
    
    # 获取第一个结果
    location = data[0]
    lat = location['lat']
    lon = location['lon']
    city_name = location.get('name', city)  # 使用API返回的城市名
    country = location.get('country', '')
    logger.info(f'城市坐标: {city_name}({country}) @ {lat},{lon}')

    # 修复2: 使用兼容的天气API (v2.5)
    # 注意: 这里改用当前天气接口，不是onecall
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    weather_resp = requests.get(weather_url)
    
    if weather_resp.status_code != 200:
        return {"error": f"天气请求失败: {weather_resp.status_code}"}
    
    weather_data = weather_resp.json()
    logger.info(f'天气数据: {weather_data}')

    # 修复3: 解析正确的v2.5响应结构
    return {
        "temperature": weather_data['main']['temp'],
        "description": weather_data['weather'][0]['description'],
        "city": city_name,
        "country": country
    }

# 工具描述 (无需放在单独模块)
WEATHER_SEARCH = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "根据城市名称获取天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如：北京"
                }
            },
            "required": ["city"]
        }
    }
}