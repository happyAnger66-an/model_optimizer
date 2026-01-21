import importlib.util
import sys
import json
import addict


class Config:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, 'r') as f:
            self.config = addict.Dict(json.load(f))


def load_settings(file_path):
    # 创建模块规范
    spec = importlib.util.spec_from_file_location("settings", file_path)

    # 创建模块
    module = importlib.util.module_from_spec(spec)

    # 将模块添加到sys.modules
#    sys.modules["settings"] = module

    # 执行模块代码
    spec.loader.exec_module(module)

    return module


if __name__ == "__main__":
    import sys
    settings = load_settings(sys.argv[1])
    print(settings.QUANT_CFG_CHOICES["int8"])
