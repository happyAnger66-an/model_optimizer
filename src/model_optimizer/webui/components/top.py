from typing import TYPE_CHECKING

import gradio as gr

from ..extras.constants import SUPPORTED_MODELS

if TYPE_CHECKING:
    from gradio.components import Component


def create_top() -> dict[str, "Component"]:
    with gr.Row():
        lang = gr.Dropdown(
            choices=["en", "ru", "zh", "ko", "ja"], value='zh', scale=1)

    with gr.Row():
        available_models = list(SUPPORTED_MODELS.keys())
        model_name = gr.Dropdown(choices=available_models, value=None, scale=2)
        model_path = gr.Textbox(label="模型路径")

    return dict(
        lang=lang,
        model_name=model_name,
        model_path=model_path
    )
