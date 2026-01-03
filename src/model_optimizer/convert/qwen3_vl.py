import os
import time
import torch


def convert_qwen3_vl(model, export_dir):
    output_dir = export_dir
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()
    print(f"Start export onnx ...")
    input_ids = torch.randint(1, model.config.vocab_size, (1, 968),
                              dtype=torch.int64,
                              device="cuda",
                              )
    attention_mask = torch.randn((1, 1, 968, 968),
                                 dtype=torch.float32,
                                 device="cuda",
                                 )
    position_ids = torch.randint(1, 1000, (1, 968),
                                 dtype=torch.int64,
                                 device="cuda",
                                 )
# am: torch.Size([1, 1, 968, 968]) - torch.float32, pi: torch.Size([1, 968])-torch.int64 prefix: torch.Size([1, 968, 2048])-torch.bfloat16
    with torch.inference_mode():
        torch.onnx.export(
            model,
            # Include position_ids in ONNX export
            (input_ids, attention_mask, position_ids),
            f"{output_dir}/qwen3_vl.onnx",
            # Add position_ids to input names
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["k_v_caches", "last_hidden_state"],
            opset_version=19,
            dynamo=False,
            do_constant_folding=True,
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "position_ids": {0: "batch_size"},
            },
        )
    end = time.time()
    print(f"export onnx to {output_dir} done cost:{end - start}s")
    return model
