import torch

def hook_module_inputs(model, hook_func, target_cls):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, target_cls):
            print(f'do register forward pre hook. name:{name} {type(module)}')
            hook = module.register_forward_pre_hook(
                hook_func, with_kwargs=True)
            hooks.append(hook)
    return hooks


def hook_module_outputs(model, hook_func, target_cls):
    hooks = []
    for name, module in model.named_modules():
        print(f'try register forward hook. name:{name} {type(module)}')
        if isinstance(module, target_cls):
            print(f'do register forward hook. name:{name} {type(module)}')
            hook = module.register_forward_hook(
                hook_func)
            hooks.append(hook)
    return hooks

def hook_module_outputs_by_name(model, hook_func, target_name):
    hooks = []
    for name, module in model.named_modules():
        if name == target_name:
            print(f'do register forward hook. name:{name} {type(module)}')
            hook = module.register_forward_hook(
                hook_func)
            hooks.append(hook)
    return hooks


class FoundationPoseOutputHook:
    def __init__(self, save_file=None):
        self.outputs = []
        self.save_file = save_file

    def start_collect(self, target_cls, target_module):
        def hook_output(m, input, output):
            print(f'output {m._get_name()}: ')
            ont_output = []
            for k, v in output.items():
#                print(f'output {k}: {type(v)} {v.shape} {v.dtype} {v.device} value:{v}')
                ont_output.append(v.clone().detach().cpu().numpy())
            self.outputs.append(ont_output)
        
        self.hooks = hook_module_outputs(target_module,
                                         hook_output, target_cls)

    def stop_collect(self):
        print(f'collect {len(self.outputs)} outputs')
        for hook in self.hooks:
            hook.remove()
        if self.save_file:
            print(f'save outputs to {self.save_file}...')
            torch.save(self.outputs, self.save_file)


class FoundationPoseCalibCollector:
    def __init__(self, save_file=None):
        self.calib_inputs = []
        self.hooks = []
        self.save_file = save_file

    def start_collect(self, target_cls, target_model):
        def hook_input(m, args, kwargs):
            print(f'hook module input: {type(m)} args:{len(args)} ')
            one_input = []
            for i, arg in enumerate(args):
                print(f'arg:{type(arg)} {arg.shape} {arg.dtype} {arg.device}')
                one_input.append(arg.clone().cpu().numpy())
            self.calib_inputs.append(one_input)

        self.hooks = hook_module_inputs(target_model,
                                        hook_input, target_cls)