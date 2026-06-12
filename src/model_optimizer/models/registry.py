ModelMaps = {}

def register_model_cls(model_name, cls):
    ModelMaps[model_name] = cls

def get_model_cls(model_name):
    if model_name not in ModelMaps:
        raise ValueError(f"Model {model_name} not found")
    return ModelMaps[model_name]

def init_registry():
    from .pi05.vit import Vit
    register_model_cls("pi05_libero/vit", Vit)
    
    from .pi05.llm import LLM
    register_model_cls("pi05_libero/llm", LLM)
    
    from .pi05.dit import Pi05DenoiseStep
    register_model_cls("pi05_libero/denoise", Pi05DenoiseStep)
init_registry()