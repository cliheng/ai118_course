from doctest import Example
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import function_tools
import requests
import gradio as gr
import random
import function_tools  # 添加导入语句

def weather_search():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info("weather_search")
    # 基本参数配置
    apiUrl = 'http://apis.juhe.cn/simpleWeather/query'  # 接口请求URL
    apiKey = '18108b0f28a5a63960cd38108aab778f'  # 在个人中心->我的数据,接口名称上方查看

    # 接口请求入参配置
    requestParams = {
        'key':apiKey,
        'city': '苏州',
    }

    # 发起接口网络请求
    response = requests.get(apiUrl, params=requestParams)

    # 解析响应结果
    if response.status_code == 200:
        responseResult = response.json()
        return f'{responseResult["result"]}'
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        print('请求异常')

def process_query(input_text, platform, model, temperature, max_tokens):
    try:
        load_dotenv()
        client = OpenAI(
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url=os.getenv("ZHIPU_API_BASE_URL")
        )
        
        tools = [function_tools.WEATHER_SEARCH]
        messages = [
            {'role':'system','content':'直接调用工具回答用户问题'},
            {'role':'user','content':input_text}
        ]

        response = client.chat.completions.create(
            model=model,
            tools=tools,
            messages=messages
        )

        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                function_name = tool_call.function.name
                invoke_fun = getattr(function_tools, function_name)
                result = invoke_fun(**args)
                messages.append({
                    'role': 'tool',
                    'content': json.dumps(result, ensure_ascii=False),
                    'tool_call_id': tool_call.id
                })
            
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )

        return response.choices[0].message.content, ""

    except Exception as e:
        return response.choices[0].message.content, 
# 创建Gradio界面
with gr.Blocks(title="天气查询助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌤️ 实战：天气查询助手（Function Calling）")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 输入部分
            input_text = gr.Textbox(label="请输入", placeholder="例如：今天北京天气怎么样？")
               # 模型回复
            model_response = gr.Textbox(
                label="模型回复", 
                value="北京今天的天气是阴天，气温为17摄氏度，湿度为82%，空气质量指数为57。",
                lines=4,
                interactive=False
            )
            
            submit_btn = gr.Button("提交", variant="primary")
        
        with gr.Column(scale=2):
         
                # 模型平台选择
            platform = gr.Radio(
                choices=["openai", "ZhipuAI", "Bailian"],
                label="大模型平台",
                value="openai"
            )
            
            # 模型选择
            model = gr.Dropdown(
                choices=["gpt-3.5-turbo", "gpt-4", "glm-4", "bailian-1.0"],
                label="模型名称",
                value="gpt-3.5-turbo"
            )
            
            # 参数设置
            temperature = gr.Slider(
                minimum=0.1, maximum=1.0, step=0.1,
                value=0.8, label="temperature"
            )
            
            max_tokens = gr.Slider(
                minimum=128, maximum=1024, step=64,
                value=512, label="max_tokens"
            )
    # 提交按钮事件
    submit_btn.click(
        fn=process_query,
        inputs=[input_text, platform, model, temperature, max_tokens],
        outputs=[model_response]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()