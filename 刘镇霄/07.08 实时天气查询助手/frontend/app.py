import gradio as gr
from backend.model_handler import main

# 后端传递数据
def query_weather(input_question, platform, model, temperature, max_tokens):
    result = main(input_question, model, temperature, max_tokens, platform)
    return result
# 动态更新下拉列表
def update_model_options(platform):
    if platform == "智谱":
        return gr.Dropdown(
            choices=["glm-4-flash", "GLM-4-Air-250414", "GLM-Z1-Flash"],
            value="glm-4-flash"
        )
    elif platform == "阿里云百炼":
        return gr.Dropdown(
            choices=["qwen-plus", "deepseek-v3"],
            value="qwen-plus"
        )
    elif platform == "API2D":
        return gr.Dropdown(
            choices=["gpt-3.5-turbo"],
            value="gpt-3.5-turbo"
        )
# 前端代码
with gr.Blocks(fill_height=True) as app:
    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            gr.Markdown("# 🌤️天气查询助手📅")
            input_question = gr.Textbox(label="请输入您的问题", lines=2)
            submit_btn = gr.Button("提交", size="lg")
            output_answer = gr.Textbox(
                label="大模型回答",
                interactive=False,
                lines=10,
                max_lines=20
            )

            gr.Examples(
                examples=[
                    "今天北京天气如何？",
                    "今天北京的空气质量？",
                    "未来北京几天的天气怎么样？"
                ],
                inputs=input_question,
                label="示例问题"
            )

        with gr.Column(scale=1):
            gr.Markdown("## 仪表盘", height=0)
            platform = gr.Radio(
                choices=["智谱", "阿里云百炼", "API2D"],
                label="大模型平台",
                value="openai"
            )
            model = gr.Dropdown(
                choices=[],
                label="模型名称",
                value="glm-4-flash"
            )
            temperature = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.7,
                label="大模型灵活度"
            )
            max_tokens = gr.Slider(
                minimum=0,
                maximum=1000,
                step=1,
                value=100,
                label="输出字数限制"
            )

    submit_btn.click(
        fn=query_weather,
        inputs=[input_question, platform, model, temperature, max_tokens],
        outputs=output_answer
    )

    platform.change(
        fn=update_model_options,
        inputs=platform,
        outputs=model
    )
if __name__ == "__main__":
    app.launch()
