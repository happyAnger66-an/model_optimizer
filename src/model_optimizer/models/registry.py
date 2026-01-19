ModelMaps = {}

def register_model_cls(model_name, cls):
    ModelMaps[model_name] = cls

def get_model_cls(model_name):
    if model_name not in ModelMaps:
        raise ValueError(f"Model {model_name} not found")
    return ModelMaps[model_name]

def init_registry():
    from .pi05.model_pi05 import Pi05Model
    register_model_cls("pi05_libero", Pi05Model)
    
    from .pi05.vit import Vit
    register_model_cls("pi05_libero/vit", Vit)
    
    from .pi05.llm import LLM
    register_model_cls("pi05_libero/llm", LLM)
    
    from .pi05.expert import Expert
    register_model_cls("pi05_libero/expert", Expert)

    from .yolo import YoloModel
    register_model_cls("yolo", YoloModel)
    
    from .yolo import YoloModel
    register_model_cls("yolo_tube", YoloModel)

init_registry()