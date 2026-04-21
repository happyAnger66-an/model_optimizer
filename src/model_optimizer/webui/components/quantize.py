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
        quantize_cfg = gr.Textbox(value=None, scale=3, label="量化配置文件路径")

    with gr.Row():
        calibrate_data = gr.Textbox(value=None, scale=3, label="校准数据路径（calibrate_data）")

    with gr.Row():
        measure_quant_error = gr.Checkbox(
            value=False,
            label="PTQ 后测量张量级 QDQ 误差（--measure-quant-error）",
        )

    with gr.Row():
        current_time = gr.Textbox(visible=False, interactive=False)
        export_dir = gr.Textbox(value=None, scale=3, label="导出目录（export_dir）")
    
    with gr.Row():
        #resume_btn = gr.Checkbox(visible=True, interactive=False)
        progress_bar = gr.Slider(visible=True, interactive=False)

    with gr.Row():
        output_box = gr.Markdown()

    input_elems.update(
        {export_dir, quantize_cfg, calibrate_data, measure_quant_error})
    elem_dict.update(dict(quantize_cfg=quantize_cfg,
                          calibrate_data=calibrate_data,
                          measure_quant_error=measure_quant_error,
                          export_dir=export_dir,
                          output_box=output_box,
        #                  resume_btn=resume_btn,
                          progress_bar=progress_bar))
    output_elems = [output_box, progress_bar]
    
    start_btn = gr.Button("开始量化")
    start_btn.click(fn=engine.runner.run_quantize, inputs=input_elems,
                        outputs=output_elems, api_name="quantize")
#    resume_btn.change(engine.runner.monitor, outputs=output_elems, concurrency_limit=None)

#    print(f'create_quantize_tab elem_dict: {elem_dict}')
    return elem_dict
