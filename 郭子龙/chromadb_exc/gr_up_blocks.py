import gradio as gr
def chat_msg(message,):
    return 'Hello,I am a chatbot!'

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(),
    tbox = gr.MultimodalTextbox(sources=["upload"],file_count="single",file_types=['image'])
    tbox.submit(chat_msg, inputs=tbox, outputs=chatbot)
demo.launch()