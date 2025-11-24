from typing import TYPE_CHECKING

import gradio as gr

from ..commom import DEFAULT_DATA_DIR

if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def quantize_model(data):
    return "成功量化了 " + f'{data}' + "!!!!"


def create_quantize_tab(engine: "Engine") -> dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        quantization_bit = gr.Dropdown(
            choices=["int8", "fp8", 'nvfp4'], value="int8", allow_custom_value=True)
        calibrate_method = gr.Dropdown(
            choices=["amax", "entroy"], value="entroy")

    with gr.Row():
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=1)
#        dataset = gr.Dropdown(
#            multiselect=True, allow_custom_value=True, scale=4)

    with gr.Row():
        #resume_btn = gr.Checkbox(visible=True, interactive=False)
        progress_bar = gr.Slider(visible=True, interactive=False)

    with gr.Row():
        current_time = gr.Textbox(visible=False, interactive=False)
        output_dir = gr.Dropdown(allow_custom_value=True)

    with gr.Row():
        output_box = gr.Markdown()

    input_elems.update(
        {output_dir, quantization_bit, calibrate_method, dataset_dir})
    elem_dict.update(dict(quantization_bit=quantization_bit, 
                          calibrate_method=calibrate_method,
                          dataset_dir=dataset_dir, 
#                          dataset=dataset,
                          output_dir=output_dir,
                          output_box=output_box,
        #                  resume_btn=resume_btn,
                          progress_bar=progress_bar))
    output_elems = [output_box, progress_bar]
    
    start_btn = gr.Button("开始量化")
    start_btn.click(fn=engine.runner.run_quantize, inputs=input_elems,
                        outputs=output_elems, api_name="quantize")
#    resume_btn.change(engine.runner.monitor, outputs=output_elems, concurrency_limit=None)

    print(f'create_quantize_tab elem_dict: {elem_dict}')
    return elem_dict
