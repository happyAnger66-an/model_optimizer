from openpi.training import config as _config
from openpi.policies import policy_config

from .model_pi05 import Pi05Model
def create_tensorrt_policy(model_name, model_path, engine_path, device="cuda"):
    config = _config.get_config(model_name)
    policy = policy_config.create_trained_policy(config, checkpoint_dir=model_path)
    pi05_model = Pi05Model(model_name, model_path, policy._model)
    pi05_model.setup_tensorrt(engine_path, device)
    return policy