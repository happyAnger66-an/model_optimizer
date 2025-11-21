import numpy as np

from model_optimizer.torch_hooks.hooks import hook_module_inputs

class YOLOCalibCollector:
    def __init__(self):
        self.calib_dict = {}
        self._datas = []
        self.hooks = []

    def start_collect(self, target_cls, target_model):
        def hook_input(m, args, kwargs):
            print(f'hook module input: {type(m)} args:{len(args)} ')
            for arg in args:
                one_input = arg.clone().cpu().numpy()
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