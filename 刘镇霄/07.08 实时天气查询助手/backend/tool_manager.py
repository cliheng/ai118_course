from backend.weather_tools.weather_parser import weather_search
# 调用工具函数的格式
WEATHER_SEARCH={
    "type": "function",
        "function": {
            "name": "weather_search_tool",
            "description": "根据用户提供的信息通过weather_search在互联网搜索引擎查询对应问题的信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京"
                    }
                },
                "required": ["city"]
            }
        }
}
# 工具函数
def weather_search_tool(city):
    return weather_search(city)