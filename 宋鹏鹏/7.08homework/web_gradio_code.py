import gradio as gr
from utils import weather_search

def chatbot_interface(query,model_type,model_name,temperature):
    response = weather_search(city=None, model=model_name, temperature=temperature, query=query)
    return response

with gr.Blocks() as demo:
    
    with gr.Row():
        gr.Markdown("# 天气查询助手")
    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(label="请输入问题",lines=6)
            submit_button = gr.Button("提交",size="lg")

        with gr.Column(scale=1):
            model_type = gr.Radio(["Openai","Zhipuai","Bailian"],label="模型选择",value="Openai")
            #添加一个下拉选项框，根据model_type,显示不同选项，如果是openai，下拉为gpt-3.5-turbo，如果是zhipuai，下拉为glm-4-flash,如果为Bailian，下拉为qwen1.5-0.5b-chat
            model_name = gr.Dropdown(choices=["gpt-3.5-turbo"], label="模型名称", value="gpt-3.5-turbo")

            def update_model_name(model_type):
                if model_type == "Openai":
                    return gr.update(choices=["gpt-3.5-turbo"], value="gpt-3.5-turbo")
                elif model_type == "Zhipuai":
                    return gr.update(choices=["glm-4-flash"], value="glm-4-flash")
                elif model_type == "Bailian":
                    return gr.update(choices=["qwen1.5-0.5b-chat"], value="qwen1.5-0.5b-chat")

            model_type.change(
                update_model_name,
                inputs=model_type,
                outputs=model_name
                )
            
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1,label="Temperature")
            

    with gr.Row():
        text_output = gr.Textbox(label="输出结果",lines=6)

    submit_button.click(
        fn=chatbot_interface,
        inputs=[query,model_type,model_name,temperature],
        outputs=[text_output]
    )
    examples = [
        ["今天北京天气怎么样？"],
        ["今天北京的空气质量如何？"],
        ["今天北京有雨吗？"]
    ]
    gr.Examples(examples,[query])

demo.launch()

