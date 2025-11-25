import os
import platform

import gradio as gr

from .engine import Engine
from .components.quantize import create_quantize_tab
from .components.eval import create_eval_tab
from .components.top import create_top
from .components.footer import create_footer

from .commom import save_config

def create_ui(demo_mode: bool = False) -> "gr.Blocks":
    engine = Engine(demo_mode=demo_mode, pure_chat=False)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]
    
    with gr.Blocks(title=f"Model Factory ({hostname})") as demo:
        title = gr.HTML()
        engine.manager.add_elems("head", {"title": title})
        engine.manager.add_elems("top", create_top())
        lang: gr.Dropdown = engine.manager.get_elem_by_id("top.lang")

        with gr.Tab("量化"):
            engine.manager.add_elems("quantize", create_quantize_tab(engine))

        with gr.Tab("评测"):
            engine.manager.add_elems("eval", create_eval_tab(engine))

        engine.manager.add_elems("footer", create_footer())
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)
        
    return demo

def run_web_ui() -> None:
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    create_ui().queue().launch(inbrowser=True)