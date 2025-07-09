import gradio as gr
from function_weather import client_openai,client_zhipu,client_qianwen, HUIJU_SEARCH
import function_weather
import json
from openai import OpenAI



def query_weather(query,model,temperature):
    if temperature is None:
        temperature = 1

    # 1. Use LLM to extract city from query
    messages = [{'role': 'user', 'content': query}]
    tools = [HUIJU_SEARCH]
    
    if model == "openai":
        client = client_openai
        model_name = "gpt-4o-mini"
    elif model == "bailian":
        client = client_qianwen
        model_name = "qwen-plus"
    elif model == "zhupu":
        client = client_zhipu
        model_name = "glm-4-flash"
    else:
        return "未知模型"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        city = None
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.function.name == 'huiju_weather_search':
                    function_args = json.loads(tool_call.function.arguments)
                    city = function_args.get("city")
                    break
        
        if not city:
            return "无法从您的提问中识别出城市名称，请问您想查询哪个城市的天气？"
    except Exception as e:
        return f"调用模型时出错: {e}"


    # 2. 先查API，获取实时天气
    result = function_weather.huiju_weather_search(city)
    if 'error' in result:
        return f"查询失败：{result['error']}"
    if 'result' in result:
        weather = result['result']
        if weather is not None and isinstance(weather, dict):
            realtime = weather.get('realtime', {})
            weather_info = json.dumps(realtime, ensure_ascii=False)
        else:
            weather_info = json.dumps({}, ensure_ascii=False)
    else:
        weather_info = json.dumps(result, ensure_ascii=False)

    # 3. 再让大模型润色输出
    prompt = f"请根据以下实时天气信息，用简洁自然的语言回答用户问题：{query}\n天气数据：{weather_info}"
    
    if model == "openai":
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    elif model == "bailian":
        response = client_qianwen.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    elif model == "zhupu":
        response = client_zhipu.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    else:
        return weather_info



with gr.Blocks() as demo:
    gr.Markdown("# 天气查询小程序（聚合数据API）")

    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(label="请输入", lines=6)

        with gr.Column(scale=1):
            model_input = gr.Radio(["zhupu", "openai", "bailian"], label="模型选择", value="openai")
            temperature = gr.Slider(minimum=0.0, maximum=1.0, label="temperatuer", value=0.8, step=0.1)
            submit_butten = gr.Button("提交", size="lg")

    with gr.Row():
        text_output = gr.Textbox(label="输出", lines=3)

    submit_butten.click(
        fn=query_weather,
        inputs=[query, model_input, temperature],
        outputs=[text_output]
    )

    examples = [
        "今天北京的天气怎么样？",
        "今天北京的空气质量怎么样？",
        "未来几天北京的空气质量怎么样？"
    ]
    gr.Examples(examples, [query])

    demo.launch()
