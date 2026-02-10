from polygraphy.comparator import Comparator, CompareFunc, DataLoader

import tensorrt as trt


class LayerOutputProfiler(trt.IProfile):
    def __init__(self):
        super().__init__()
        self.layers_outputs = {}
        self.layer_times = {}