def hook_module_inputs(model, hook_func, target_cls):
    hooks = []
    for name, module in model.named_modules():
        print(f'register forward pre hook. name:{name} {type(module)}')
        if isinstance(module, target_cls):
#            print(f'do register forward pre hook. name:{name} {type(module)}')
            hook = module.register_forward_pre_hook(
                hook_func, with_kwargs=True)
            hooks.append(hook)
    return hooks


def hook_module_outputs(model, hook_func, target_cls):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, target_cls):
#            print(f'do register forward hook. name:{name} {type(module)}')
            hook = module.register_forward_hook(
                hook_func)
            hooks.append(hook)
    return hooks
