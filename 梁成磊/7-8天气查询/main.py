import streamlit as st
import llm


def main():
    st.set_page_config(layout="wide")
    st.title("实战：实时天气查询助手（Function Calling）")

    if "example" not in st.session_state:
        st.session_state.example = None
    if "response" not in st.session_state:
        st.session_state.response = ""

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.container():
            question = st.text_area(
                label="请输入", height=255, value=st.session_state.example
            )
            click = st.button("提交", use_container_width=True)

    with col2:
        with st.container():
            model_platform = st.radio(
                label="大模型平台",
                options=["OpenAI", "ZhipuAI", "BaiLian"],
                horizontal=True,
            )
            model_list = []
            if model_platform == "OpenAI":
                model_list = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
            elif model_platform == "ZhipuAI":
                model_list = ["glm-4-flash-250414", "glm-4-air-250414", "glm-4-plus"]
            else:
                model_list = ["qwen-turbo", "qwen-plus-latest", "qwen-turbo-latest"]
            model = st.selectbox(label="模型名称", options=model_list)
            temperature = st.slider(
                "temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            )
            max_tokens = st.slider(
                "max_tokens", min_value=1, max_value=2048, value=1024
            )

    with st.container():
        st.text_area(
            label="模型回复", value=st.session_state.response, disabled=True, height=240
        )

    if click:
        if model_platform == "OpenAI":
            response = llm.openai_llm(model, temperature, max_tokens, question)
            if response and st.session_state.response != response:
                st.session_state.response = response
        elif model_platform == "ZhipuAI":
            response = llm.zhipu_llm(model, temperature, max_tokens, question)
            if response and st.session_state.response != response:
                st.session_state.response = response
        else:
            response = llm.bailian_llm(model, temperature, max_tokens, question)
            if response and st.session_state.response != response:
                st.session_state.response = response
                
        st.session_state.example = None
        st.rerun()

    with st.container():
        example = st.pills(
            label="Examples",
            options=[
                "今天北京天气怎么样",
                "今天北京的空气质量如何",
                "未来几天北京的天气怎么样",
            ],
            selection_mode="single",
        )

        if example and st.session_state.example != example:
            st.session_state.example = example
            st.rerun()


if __name__ == "__main__":
    main()
