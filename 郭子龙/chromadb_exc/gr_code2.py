import gradio as gr

# gr.load_chat("http://localhost:11434/v1/", model="deepseek-r1:latest ", token="***").launch()
demo = gr.load_chat(
    "https://open.bigmodel.cn/api/paas/v4/",
    model="glm-4-air-250414 ",
    token="59251781bb444688a4656508dceccaca.Lo7JWjD9N0XKfIu8",
    system_message='你是一名经验丰富的大学历史老师，能够回答关于历史的一切问题'
)
if __name__ == "__main__":
    demo.launch(share=True)