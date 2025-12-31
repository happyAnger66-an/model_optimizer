from typing import TYPE_CHECKING

import gradio as gr

from ..commom import DEFAULT_DATA_DIR

if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_export_tab(engine: "Engine") -> dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        export_format = gr.Dropdown(
            choices=["onnx"], value="onnx", allow_custom_value=True)
    
    with gr.Row():
        simplifier = gr.Checkbox(value=True)

    with gr.Row():
        output_dir = gr.Dropdown(allow_custom_value=True)
    
    with gr.Row():
        progress_bar = gr.Slider(visible=True, interactive=False)

    with gr.Row():
        output_box = gr.Markdown()

    input_elems.update(
        {output_dir, export_format, simplifier})
    elem_dict.update(dict(export_format=export_format, 
                          simplifier=simplifier,
                          output_dir=output_dir,
                          output_box=output_box,
                          progress_bar=progress_bar))
    output_elems = [output_box, progress_bar]
    
    start_btn = gr.Button("开始导出")
    start_btn.click(fn=engine.runner.run_export, inputs=input_elems,
                        outputs=output_elems, api_name="export")

    return elem_dict
