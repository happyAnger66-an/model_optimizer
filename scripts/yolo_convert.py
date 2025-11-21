import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    # load a pretrain model
    model = YOLO(args.model_path)
    print(f' load model {model}')
   
    model.export(format="onnx", dynamic=True, simplify=True)

    print(f'end export model...')