import Weather_requst
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import gradio as gr


def requests(query,model_input):
    load_dotenv()

    tools = [Weather_requst.OPEN_WEATHER]

    messages = [
        {'role': 'system','content': '根据用户提问的地点，调用天气API得到对应信息后输出当地天气情况'},
        {"role": "user", "content": query}
    ]
    if model_input == 'gpt-3.5-turbo':
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url=os.environ['BASE_URL'])
    elif model_input == 'glm-4-plus':
        client = OpenAI(api_key=os.environ['ZHIPU_API_KEY'],base_url=os.environ['ZHIPU_BASE_URL'])
    elif model_input == 'deepseek-chat':
        client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],base_url=os.environ['DEEPSEEK_BASE_URL'])
    resp = client.chat.completions.create(
        model=model_input,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    # print(resp)
    while resp.choices[0].message.tool_calls is not None:
        messages.append(resp.choices[0].message)
        for tool_call in resp.choices[0].message.tool_calls:
            # 将回答中toolname保存
            function_name = tool_call.function.name
            # 将回答中arguments参数保存
            argument = tool_call.function.arguments
            arguments = json.loads(argument) #转为字典
            # 外部调用，在function_tool中找function_name
            invoke = getattr(Weather_requst, function_name)
            res = invoke(**arguments)
            # print(res)
            messages.append({
                        'role': 'tool',
                        'content': json.dumps(res), #转为json
                        'tool_call_id': tool_call.id
            })
        # 第二次调用，将最终结果返回
        resp = client.chat.completions.create(
            model=model_input,
            messages=messages,
            tools=tools,
            tool_choice='auto'
        )
    return resp.choices[0].message.content

def chatbot_interface(query,model_input,temperature):
    # response = requests(query,'glm-4-plus')
    # print(response)
    if model_input == 'OpenAI':
        response = requests(query,'gpt-3.5-turbo')
    elif model_input == 'ZhipuAI':
        response = requests(query,'glm-4-plus')
    elif model_input == 'DeepseekAI':
        response = requests(query,'deepseek-chat')
    return response
if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('实战：天气查询助手（Function_calling）')
        with gr.Row():
            text_output = gr.Textbox(label='模型回复',lines=9)
        with gr.Row():
            with gr.Column(scale=2):
                query = gr.Textbox(label='请输入需要查询天气的城市',lines=6)
            with gr.Column():
                model_input = gr.Radio(['OpenAI','ZhipuAI','DeepseekAI'],label='模型',value='openai')
                # temperature = gr.Slider(0,1,value=0.8,label='temperature',step=0.1)
                submit_button = gr.Button('提交',size='lg')
        #按钮方法，触发函数，传入input，在输出框输出output
        submit_button.click(
            fn = chatbot_interface,
            inputs=[query,model_input],
            outputs=[text_output]
        )    

        examples = [
            ['今天北京天气怎么样？'],
            ['今天北京空气质量天气怎么样？'],
            ['未来几天北京天气怎么样？']
        ]
        gr.Examples(examples,[query])
    demo.launch()    

