import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import function_tools
from zhipuai import ZhipuAI




# 模型配置映射表
# 1. 修改模型映射表
MODEL_CONFIGS = {
    "openai": {
        "client": OpenAI,
        "api_key": lambda: os.getenv("OPENAI_API_KEY"),  # 改为延迟加载
        "base_url": lambda: os.getenv("OPENAI_API_BASE")
    },
    "ZhipuAI": {
        "client": ZhipuAI,
        "api_key": lambda: os.getenv("ZHIPUAI_API_KEY"),
        "base_url": lambda: os.getenv("ZHIPUAI_API_BASE")
    }
    # 暂时移除Bailian配置直到实现
}

# 2. 增强模型识别逻辑
def get_platform(model_name):
    model_map = {
        'gpt': 'openai',
        'chatglm': 'ZhipuAI',
        'glm': 'ZhipuAI'
    }
    prefix = model_name.split('-')[0].lower()
    return model_map.get(prefix, 'openai')  # 默认fallback

# 3. 修改客户端初始化
# 修改客户端初始化部分
def init_client(model_name):
    platform = get_platform(model_name)
    if platform not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported platform: {platform}")
    
    config = MODEL_CONFIGS[platform]
    api_key = config["api_key"]() if callable(config["api_key"]) else config["api_key"]
    base_url = config["base_url"]() if callable(config["base_url"]) else config.get("base_url")
    
    return config["client"](
        api_key=api_key,  # 确保传递的是字符串值
        base_url=base_url  # 确保传递的是字符串值
    )

if __name__ == "__main__":
    load_dotenv()
    
    # 修复默认客户端初始化
    config = MODEL_CONFIGS["openai"]
    client = config["client"](
        api_key=config["api_key"](),  # 调用lambda获取实际值
        base_url=config["base_url"]() if config.get("base_url") else None
    )

def chatbot_interface(query, model_select, temperature):
    tools = [function_tools.WEATHER_SEARCH]
    messages = [
        {"role": "system", "content": "不需要用户补充问题，直接调用tool"},
        {"role": "user", "content": query}
    ]
    
    current_client = init_client(model_select)
    
    try:
        response = current_client.chat.completions.create(
            model=model_select,
            messages=messages,
            tools=tools,
            temperature=temperature
        )
        
        while hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            tool_message = response.choices[0].message
            # 修复：正确序列化工具调用
            tool_calls = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments
                    }
                } for call in tool_message.tool_calls
            ]
            
            messages.append({
                "role": tool_message.role,
                "content": tool_message.content or "",
                "tool_calls": tool_calls
            })
            
            for tool_call in tool_message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                function_name = tool_call.function.name
                result = getattr(function_tools, function_name)(**args)
                
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id
                })
            
            response = current_client.chat.completions.create(
                model=model_select,
                messages=messages,
                tools=tools
            )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# 实战：天气查询助手（Function Calling）")

    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(label="请输入", lines=6)
        with gr.Column(scale=1):
            model_input = gr.Radio(["openai", "ZhipuAI", "Bailian"], label="模型", value="openai")

            model_select = gr.Dropdown(
                    ["gpt-3.5-turbo", "gpt-4"], 
                    label="模型选择",
                    value="gpt-3.5-turbo",
                    interactive=True
                )
            max_tokens = gr.Slider(1, 2048, value=100, label="max_tokens", step=1, interactive=True)
            temperature = gr.Slider(minimum=0.0, maximum=1.0, label="temperature", value=0.8, step=0.1)
            submit_button = gr.Button("提交", size="lg")
    def update_models(model_input):
        models = {
            "openai": ["gpt-3.5-turbo", "gpt-4"],
            "ZhipuAI": ["chatglm_std", "glm-4-flash"],
            "Bailian": ["baichuan2-7b", "baichuan2-13b"]
        }
        return gr.Dropdown(choices=models[model_input], value=models[model_input][0])
    model_input.change(update_models, inputs=model_input, outputs=model_select)

    with gr.Row():
        text_output = gr.Textbox(label="模型回复", lines=3)

    # 定义按钮点击事件的回调函数
    submit_button.click(
        fn=chatbot_interface,
        inputs=[query, model_select, temperature],
        outputs=[text_output]
    )

    # 添加 Examples 组件
    examples = [
        ["今天北京天气怎么样？"],
        ["今天北京的空气质量如何？"],
        ["未来几天北京的天气怎么样？"]
    ]
    gr.Examples(examples, [query])

demo.launch()