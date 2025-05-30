#这里目前有两个参数大模型的回答answer，用户的问题question,实现history，聊天记录的保存，还有回复用户问题

import gradio as gr

def chat_gradio(answer_func):
    def echo(message, history):
        answer = answer_func(message)
        # 返回openai风格格式，兼容type='messages'
        return {"role": "assistant", "content": answer}

    demo = gr.ChatInterface(
        fn=echo,
        title="随身python老师",
        description="随身Python老师，随时随地为您解答Python编程问题",
        examples=[
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好,我是你的随身python老师，赶紧跟我一起学习python吧。"}
            ]
        ],
        type="messages"  # 明确指定type，避免警告
    )
    demo.launch()
