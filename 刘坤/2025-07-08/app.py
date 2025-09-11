import streamlit as st
from function_qweather import get_weather_response

st.title("天气查询应用")

user_query = st.text_input("请输入城市名称：", "北京")

if st.button("查询天气"):
    if user_query:
        with st.spinner("正在查询天气..."):
            weather_result = get_weather_response(user_query)
            st.success("查询完成！")
            st.write(weather_result)
    else:
        st.warning("请输入城市名称。")