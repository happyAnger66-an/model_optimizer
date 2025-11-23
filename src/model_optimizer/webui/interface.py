import os
import platform

import gradio as gr

from .css import CSS

def quantize_model(name, model_path):
    return "成功量化了 " + name + "!!!!"

def create_ui(demo_mode: bool = False) -> "gr.Blocks":
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]
    
    with gr.Blocks(title=f"Model Factory ({hostname})") as demo:
        title = gr.HTML()
        subtitle = gr.HTML()

        name = gr.Textbox(label="模型")
        model_path = gr.Textbox(label="模型路径")
        output = gr.Textbox(label="量化结果")
        start_btn = gr.Button("开始量化")
        start_btn.click(fn=quantize_model, inputs=[name, model_path], outputs=output, api_name="quantize")

    return demo

def run_web_ui() -> None:
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    create_ui().queue().launch(inbrowser=True)