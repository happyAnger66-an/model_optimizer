import numpy as np

from model_optimizer.torch_hooks.hooks import hook_module_inputs


class YOLOCalibCollector:
    def __init__(self, target_shape=None):
        self.calib_dict = {}
        self._datas = []
        self._target_shape = target_shape
        self.target_shape = None
        if self._target_shape:
            self.target_shape = [int(x) for x in self._target_shape.split('x')]
        self.hooks = []

    def _shape_equal(self, input_shapes):
        if self.target_shape is None:
            return True

        if len(self.target_shape) != len(input_shapes):
            return False

        for i in range(len(self.target_shape)):
            if self.target_shape[i] != input_shapes[i]:
                return False
        return True

    def start_collect(self, target_cls, target_model):
        def hook_input(m, args, kwargs):
#            print(f'hook module input: {type(m)} args:{len(args)} ')
            for arg in args:
                one_input = arg.clone().cpu().numpy()
#                print(f'one_input shape: {one_input.shape}')
                if self._shape_equal(one_input.shape):
                    print(f'one_input shape: {one_input.shape} equal to target shape: {self.target_shape}')
                    self._datas.append(one_input)

        self.hooks = hook_module_inputs(target_model,
                                        hook_input, target_cls)

    def stop_collect(self):
        #        for data in self._datas:
        #            print(f"{type(data)} {data.shape}")
        self.calib_dict["images"] = np.concatenate(self._datas[1:])
        print(f'collect {len(self.calib_dict["images"])} inputs')
        for hook in self.hooks:
            hook.remove()

    @property
    def datas(self):
        return self.calib_dict


class SamCalibCollector(YOLOCalibCollector):
    def __init__(self):
        super().__init__()
