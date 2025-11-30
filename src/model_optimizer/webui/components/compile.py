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


def create_compile_tab(engine: "Engine") -> dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        shapes = gr.Textbox(value=None, scale=2)

    input_elems.update({shapes})
    elem_dict.update(dict(shapes=shapes))

    with gr.Row():
        do_perf = gr.Checkbox(value=True)
        extra_compile_args = gr.Textbox(value=None, scale=3)

    input_elems.update({do_perf, extra_compile_args})
    elem_dict.update(dict(do_perf=do_perf, extra_compile_args=extra_compile_args))

    with gr.Row():
        output_dir = gr.Textbox()

    input_elems.update({output_dir})
    elem_dict.update(dict(output_dir=output_dir))

    with gr.Row():
        start_btn = gr.Button(variant="primary")

    with gr.Row():
        progress_bar = gr.Slider(visible=False, interactive=False)

    with gr.Row():
        output_box = gr.Markdown()

    elem_dict.update(
        dict(
            start_btn=start_btn,
            progress_bar=progress_bar,
            output_box=output_box,
        )
    )
    output_elems = [output_box, progress_bar]

    start_btn.click(engine.runner.run_compile, input_elems, output_elems)

    return elem_dict
