from ..models.pi05.model_pi05 import Pi05Model


def convert_pi05_model(args, model_name, model_type):
    model_path, export_dir = args.model_path, args.export_dir
    pi05_model = Pi05Model(model_name, model_path)
    pi05_model.load()
    pi05_model.export_onnx(export_dir)
