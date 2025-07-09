import streamlit as st
import function_weather
import json

def query_weather(city):
    result = function_weather.huiju_weather_search(city)
    if 'error' in result:
        return f"查询失败：{result['error']}"
    if 'result' in result:
        weather = result['result']
        realtime = weather.get('realtime', {})
        info = f"城市：{city}\n温度：{realtime.get('temperature', '未知')}℃\n天气：{realtime.get('info', '未知')}\n湿度：{realtime.get('humidity', '未知')}%\n风向：{realtime.get('direct', '未知')}"
        return info
    return json.dumps(result, ensure_ascii=False)

st.title('天气查询小程序（聚合数据API）')
city = st.text_input('请输入城市名称', '北京')
if st.button('查询天气'):
    weather_info = query_weather(city)
    st.text(weather_info)
