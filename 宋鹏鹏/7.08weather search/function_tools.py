import requests
WEATHER_SEARCH={
            "type": "function",  
            "function": {
                "name": "weather_search", 
                "description": "查询指定城市的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市"
                        }
                    }
                    },
                    "required": ["city"]  
                }
            }

def weather_search(city):
    apiUrl = 'http://apis.juhe.cn/simpleWeather/query'  # 接口请求URL
    apiKey = '330cfa6a01f96d43d5e9e6b81ecafdd6'  # 在个人中心->我的数据,接口名称上方查看

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
        return f'{responseResult['result']}'
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        print('请求异常')