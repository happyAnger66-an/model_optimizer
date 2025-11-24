# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_eval_tab(engine: "Engine") -> dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        dataset_dir = gr.Textbox(value=None, scale=2)

    input_elems.update({dataset_dir})
    elem_dict.update(dict(dataset_dir=dataset_dir))

    with gr.Row():
        predict = gr.Checkbox(value=True)

    input_elems.update({predict})
    elem_dict.update(dict(predict=predict))

    with gr.Row():
        output_dir = gr.Textbox()

    input_elems.update({output_dir})
    elem_dict.update(dict(output_dir=output_dir))

    with gr.Row():
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        resume_btn = gr.Checkbox(visible=False, interactive=False)
        progress_bar = gr.Slider(visible=False, interactive=False)

    with gr.Row():
        output_box = gr.Markdown()

    elem_dict.update(
        dict(
            start_btn=start_btn,
            stop_btn=stop_btn,
            resume_btn=resume_btn,
            progress_bar=progress_bar,
            output_box=output_box,
        )
    )
    output_elems = [output_box, progress_bar]

    start_btn.click(engine.runner.run_eval, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort)
    resume_btn.change(engine.runner.monitor, outputs=output_elems, concurrency_limit=None)

    return elem_dict
