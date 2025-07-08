import gradio as gr

def chatbot_interface(query,model_input,temperature,top_p):
    response = ""
    #选择模型
    return response 

#创建gradio界面

with gr.Blocks() as demo:

    with gr.Row():
        gr.Markdown("#实战：天气查询助手（Function Calling）")

    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(label="请输入",lines=6)
    
        with gr.Column(scale=1):
            model_input = gr.Radio(["openai","ZhipuAi","Bailian"],label="模型",value="openai")
        temperature = gr.Slider(minimum=0.0,maximum=1.0,label="temperature",value=0.8,step=0.1)
        submit_button = gr.Button('提交',size="lg")

    with gr.Row():
        text_output = gr.Texbox(label="模型回复",lines=3)

    #定义按钮点击事件的回调函数
    submit_button.click(
        fn=chatbot_interface,
        inputs=[query,model_input,temperature],
        outputs=[text_output]
    )

    #添加 Examples 组件

    examples=[
        ["今天天气怎么样"],
        ["北京天气怎么样"],
        ["上海天气怎么样"]
    ]
    gr.Examples(examples,[query])

demo.launch()