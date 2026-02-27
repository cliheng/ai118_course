import requests
import logging

logging.basicConfig(level=logging.INFO)


# 天气查询
def weather_search(city):
    logging.info(f"weather_search：查询城市 : {city}")
    apiUrl = "http://apis.juhe.cn/simpleWeather/query"  # 接口请求URL
    apiKey = "5b8058322ba1e90c923c94341fb87a68"  # 在个人中心->我的数据,接口名称上方查看

    # 接口请求入参配置
    requestParams = {
        "key": apiKey,
        "city": city,
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
        raise ValueError(
            f"错误代码为{response.status_code}，错误原因为{response.reason}"
        )


# 空气质量查询
def cityair_search(city):
    logging.info(f"cityair_search：查询城市 : {city}")
    # 基本参数配置
    apiUrl = "http://web.juhe.cn/environment/air/cityair"  # 接口请求URL
    apiKey = "db762ca9452f585a5cd9cb3a16d0d63e"  # 在个人中心->我的数据,接口名称上方查看

    # 接口请求入参配置
    requestParams = {
        "key": apiKey,
        "city": city,
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
        raise ValueError(
            f"错误代码为{response.status_code}，错误原因为{response.reason}"
        )
