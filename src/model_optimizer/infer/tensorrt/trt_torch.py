# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import ctypes
import os
import time
import numpy as np

import tensorrt as trt
import torch

from transformers.modeling_outputs import BaseModelOutputWithPooling

from termcolor import colored


def torch_type(name, trt_type):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.int64: torch.int64,
        trt.bfloat16: torch.bfloat16,
    }
    if trt_type in mapping:
        return mapping[trt_type]

    raise TypeError(
        f"Could not resolve {name} TensorRT datatype to an equivalent numpy datatype. {trt_type}"
    )


def _trt_shape_has_dynamic_dims(shape: tuple) -> bool:
    """TensorRT 用负数（常为 -1）表示动态维；不能与 torch.Size 直接相等比较。"""
    return any(int(d) < 0 for d in shape)


class Engine(object):
    def __init__(
        self,
        file,
        return_wrap=None,
        perf=False,
        plugins=[],
        use_cuda_graph=None,
        cuda_graph_warmup=1,
    ):
        super().__init__()

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        self.plugins = [ctypes.CDLL(plugin, ctypes.RTLD_GLOBAL)
                        for plugin in plugins]
        self.file = file
        self.load(file)
        self.return_wrap = return_wrap
        self.perf = perf
        self.count = 0
        if use_cuda_graph is None:
            use_cuda_graph = os.getenv("MODEL_OPT_TRT_CUDA_GRAPH", "0") in ("1", "true", "True")
        self.use_cuda_graph = bool(use_cuda_graph and torch.cuda.is_available())
        self.cuda_graph_warmup = max(int(cuda_graph_warmup), 0)
        self._graph_cache = {}

        if self.perf:
            self.time_results = {
                'total': [],
            }

        def destroy(self):
            del self.execution_context
            del self.handle

        atexit.register(destroy, self)
        self.print()

    def print(self):
        if int(os.getenv("LOCAL_RANK", -1)) not in [0, -1]:
            return

        print("============= TRT Engine Detail =============")
        print(f"Engine file: {self.file}")
        print(f"CUDA Graph: {self.use_cuda_graph} (warmup={self.cuda_graph_warmup})")
        print(f"Inputs: {len(self.in_meta)}")
        for ib, item in enumerate(self.in_meta):
            tensor_name, shape, dtype = item[:3]
            print(
                f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")

        print(f"Outputs: {len(self.out_meta)}")
        for ib, item in enumerate(self.out_meta):
            tensor_name, shape, dtype = item[:3]
            print(
                f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")
        print("=============================================")

    def load(self, file):
        runtime = trt.Runtime(self.logger)

        with open(file, "rb") as f:
            self.handle = runtime.deserialize_cuda_engine(f.read())
            assert (
                self.handle is not None
            ), f"Failed to deserialize the cuda engine from file: {file}"

        self.execution_context = self.handle.create_execution_context()
        self.meta, self.in_meta, self.out_meta = [], [], []
        for tensor_name in self.handle:
            shape = self.handle.get_tensor_shape(tensor_name)
            dtype = torch_type(
                tensor_name, self.handle.get_tensor_dtype(tensor_name))
            if self.handle.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.in_meta.append([tensor_name, shape, dtype])
            else:
                self.out_meta.append([tensor_name, shape, dtype])

    def __call__(self, *args, **inputs):
        return self.forward(*args, **inputs)

    def set_runtime_tensor_shape(self, name, shape):
        self.execution_context.set_input_shape(name, shape)

    def _prepare_input_tensor(self, name, x, dtype):
        """准备单个输入：动态维时先 set_input_shape，必要时 cast 到引擎 dtype。"""
        runtime_shape = self.execution_context.get_tensor_shape(name)
        assert isinstance(x, torch.Tensor), f"Unsupported tensor type for {name}: {type(x)}"
        if _trt_shape_has_dynamic_dims(runtime_shape):
            self.execution_context.set_input_shape(name, tuple(x.shape))
        else:
            assert tuple(x.shape) == tuple(runtime_shape), (
                f"Invalid input[{name}] shape: {tuple(x.shape)} != {tuple(runtime_shape)}"
            )
        if dtype != x.dtype:
            print(colored(
                f"CastType tensor dtype {name}, excepted dtype is {dtype}, but got {x.dtype} Convert to {dtype}", "yellow"))
            x = x.to(dtype)
        assert x.is_cuda, f"Invalid tensor device for {name}, expected cuda, got {x.device}"
        return x.cuda().contiguous()

    def _collect_inputs(self, args, kwargs):
        inputs = {}
        for iarg, x in enumerate(args):
            if iarg >= len(self.in_meta):
                raise ValueError(
                    f"Too many positional inputs: got {len(args)}, expected <= {len(self.in_meta)}"
                )
            name = self.in_meta[iarg][0]
            inputs[name] = x
        for name, x in kwargs.items():
            if name == "return_list":
                continue
            inputs[name] = x
        return inputs

    def _signature_key(self, prepared_inputs):
        sig = []
        for name, _, _ in self.in_meta:
            x = prepared_inputs[name]
            sig.append((name, tuple(x.shape), x.dtype, x.device))
        return tuple(sig)

    def forward(self, *args, **kwargs):
        self.count += 1
        start_time = time.perf_counter()
        return_list = kwargs.pop("return_list", False)
        all_inputs = self._collect_inputs(args, kwargs)
        missing = [name for name, _, _ in self.in_meta if name not in all_inputs]
        if missing:
            raise ValueError(f"Missing required TRT inputs: {missing}")

        prepared_inputs = {}
        for name, _, dtype in self.in_meta:
            prepared_inputs[name] = self._prepare_input_tensor(name, all_inputs[name], dtype)

        outputs = None
        if self.use_cuda_graph:
            key = self._signature_key(prepared_inputs)
            entry = self._graph_cache.get(key)
            if entry is None:
                static_inputs = {name: torch.empty_like(prepared_inputs[name]) for name, _, _ in self.in_meta}

                for name, _, _ in self.in_meta:
                    self.execution_context.set_tensor_address(name, static_inputs[name].data_ptr())

                static_outputs = {}
                for out_name, _, out_dtype in self.out_meta:
                    runtime_shape = self.execution_context.get_tensor_shape(out_name)
                    if _trt_shape_has_dynamic_dims(runtime_shape):
                        raise RuntimeError(
                            f"Output {out_name} still has dynamic shape {runtime_shape} after binding inputs; "
                            "check that all dynamic inputs were provided."
                        )
                    static_outputs[out_name] = torch.empty(
                        tuple(runtime_shape),
                        dtype=out_dtype,
                        device=static_inputs[self.in_meta[0][0]].device,
                    )
                    self.execution_context.set_tensor_address(out_name, static_outputs[out_name].data_ptr())

                graph_stream = torch.cuda.Stream(device=static_inputs[self.in_meta[0][0]].device)
                for _ in range(self.cuda_graph_warmup):
                    with torch.cuda.stream(graph_stream):
                        for name, _, _ in self.in_meta:
                            static_inputs[name].copy_(prepared_inputs[name], non_blocking=True)
                        self.execution_context.execute_async_v3(graph_stream.cuda_stream)
                graph_stream.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_stream):
                    self.execution_context.execute_async_v3(graph_stream.cuda_stream)

                entry = {
                    "graph": graph,
                    "static_inputs": static_inputs,
                    "static_outputs": static_outputs,
                }
                self._graph_cache[key] = entry

            for name, _, _ in self.in_meta:
                entry["static_inputs"][name].copy_(prepared_inputs[name], non_blocking=True)
            entry["graph"].replay()
            torch.cuda.current_stream().synchronize()
            outputs = {name: tensor.clone() for name, tensor in entry["static_outputs"].items()}
        else:
            stream = torch.cuda.current_stream()
            reference_tensors = []
            for name, _, _ in self.in_meta:
                x = prepared_inputs[name]
                self.execution_context.set_tensor_address(name, x.data_ptr())
                reference_tensors.append(x)

            for out_name, _, out_dtype in self.out_meta:
                runtime_shape = self.execution_context.get_tensor_shape(out_name)
                if _trt_shape_has_dynamic_dims(runtime_shape):
                    raise RuntimeError(
                        f"Output {out_name} still has dynamic shape {runtime_shape} after binding inputs; "
                        "check that all dynamic inputs were provided."
                    )
                output_tensor = torch.zeros(
                    tuple(runtime_shape), dtype=out_dtype, device=reference_tensors[0].device
                )
                self.execution_context.set_tensor_address(out_name, output_tensor.data_ptr())
                reference_tensors.append(output_tensor)

            self.execution_context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            assert len(reference_tensors) == len(self.in_meta) + len(
                self.out_meta
            ), f"Invalid input tensors. The expected I/O tensors are {len(self.in_meta) + len(self.out_meta)}, but got {len(reference_tensors)}"
            outputs = {
                item[0]: reference_tensors[len(self.in_meta) + i]
                for i, item in enumerate(self.out_meta)
            }

        end_time = time.perf_counter()
        if self.perf and self.count > 100:
            self.time_results['total'].append(end_time - start_time)
            #print(colored(
            #    f"total time: {np.mean(self.time_results['total'])*1000:.2f} ± {np.std(self.time_results['total'])*1000:.2f} ms", "green"))

        if return_list:
            print(f"return_list: {return_list}")
            return [outputs[item[0]] for item in self.out_meta]
        else:
            #            output = BaseModelOutputWithPooling(
            #                last_hidden_state=reference_tensors[len(self.in_meta)]
            #            )
            #            print(f"output: {output}")
            #           return output
            output = outputs
            if self.return_wrap:
                output = self.return_wrap(output)
            return output
