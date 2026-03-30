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
    
    from .pi05.llm_with_trtedgellm import LLMWithTrtEdgeLLM
    register_model_cls("pi05_libero/llm_with_trtedge", LLMWithTrtEdgeLLM)
    
    from .pi05.expert import Expert
    register_model_cls("pi05_libero/expert", Expert)

    from .pi05.dit import Pi05DenoiseStep
    register_model_cls("pi05_libero/denoise", Pi05DenoiseStep)

    from .pi05.embed_prefix import Pi05EmbedPrefix
    register_model_cls("pi05_libero/embed_prefix", Pi05EmbedPrefix)

    from .yolo import YoloModel
    register_model_cls("yolo", YoloModel)
    
    from .yolo import YoloModel
    register_model_cls("yolo_tube", YoloModel)

init_registry()