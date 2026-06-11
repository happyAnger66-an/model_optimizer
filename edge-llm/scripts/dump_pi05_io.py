# SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""Dump pi05 TRT 管线的输入与各 stage golden 输出，供 C++ ``pi05_compare`` 对拍。

golden 取自 *Python TRT 全挂载路径*（与 C++ 用同一组 engine），因此对拍阈值可以收得很紧。

用法::

    python scripts/dump_pi05_io.py \\
        --checkpoint /path/to/ckpt --config-name pi05_libero \\
        --engine-path /path/to/engines \\
        --embed-prefix-engine embed_prefix.engine \\
        --llm-engine llm.engine --denoise-engine denoise.engine \\
        --out-dir /tmp/pi05_io [--prompt "..."] [--seed 0] [--num-steps 10] \\
        [--adarms-precompute] [--example-npz obs.npz]

输出文件（C++ 侧约定，均为 npy）::

    输入:  image_{i}.npy image_mask_{i}.npy lang_tokens.npy lang_masks.npy noise.npy
           [adarms_mod_step{k}.npy]
    golden: prefix_embs.npy prefix_pad_masks.npy past_keys.npy past_values.npy
            v_t_step{k}.npy actions.npy
    原始观测（pi05_infer 端到端用）:
           base_image.npy wrist_image.npy state.npy prompt.txt
    端到端 golden:
           actions_final.npy  （Unnormalize + LiberoOutputs 后的 [horizon, 7]）
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--engine-path", required=True)
    p.add_argument("--embed-prefix-engine", default="embed_prefix.engine")
    p.add_argument("--llm-engine", default="llm.engine")
    p.add_argument("--denoise-engine", default="denoise.engine")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompt", default="pick up the black bowl and place it on the plate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument(
        "--adarms-precompute",
        action="store_true",
        help="denoise 引擎为 AdaRMS 预计算变体（输入 adarms_mod），同时 dump 每步 modulation",
    )
    p.add_argument(
        "--example-npz",
        default="",
        help="可选：真实 obs（npz，含 observation/image, observation/wrist_image, "
        "observation/state）；缺省用固定随机合成样本",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_example(args: argparse.Namespace):
    import numpy as np

    if args.example_npz:
        data = np.load(args.example_npz, allow_pickle=True)
        example = {k: data[k] for k in data.files}
        example.setdefault("prompt", args.prompt)
        return example

    rng = np.random.default_rng(args.seed)
    return {
        "observation/image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal(8).astype(np.float32),
        "prompt": args.prompt,
    }


def to_npy(t):
    """torch tensor → numpy（bf16/fp16 golden 统一转 fp32 落盘）。"""
    import numpy as np
    import torch

    t = t.detach()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    return np.ascontiguousarray(t.cpu().numpy())


def main() -> int:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import addict
    import numpy as np
    import torch

    from model_optimizer.infer.tensorrt.pi05_executor import Pi05TensorRTExecutor
    from model_optimizer.policies import get_policy_adapter

    # ---- 加载 policy 并挂载 TRT 引擎（与 webui tensorrt 模式相同路径） ----
    adapter = get_policy_adapter("pi05")
    train_cfg = adapter.load_train_config(args.config_name)
    policy = adapter.create_policy(train_cfg, args.checkpoint, pytorch_device=args.device)

    executor = Pi05TensorRTExecutor(policy, torch.bfloat16)
    trt_cfg = {
        "engine_path": args.engine_path,
        "embed_prefix_engine": args.embed_prefix_engine,
        "llm_engine": args.llm_engine,
        "denoise_engine": args.denoise_engine,
    }
    if args.adarms_precompute:
        trt_cfg["denoise_adarms_precompute"] = True
    executor.load_model(addict.Dict(trt_cfg))

    model = executor.pi05_model
    captured: dict[str, object] = {}

    # ---- hook embed_prefix：捕获输入（预处理后图像/掩码/tokens）与输出 ----
    orig_embed_prefix = model.embed_prefix

    def embed_prefix_capture(images, img_masks, lang_tokens, lang_masks):
        out = orig_embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if "prefix_embs" not in captured:
            captured["images"] = [t.clone() for t in images]
            captured["img_masks"] = [t.clone() for t in img_masks]
            captured["lang_tokens"] = lang_tokens.clone()
            captured["lang_masks"] = lang_masks.clone()
            captured["prefix_embs"] = out[0].clone()
            captured["prefix_pad_masks"] = out[1].clone()
        return out

    model.embed_prefix = embed_prefix_capture

    # ---- hook denoise_step：捕获 KV / 每步 v_t ----
    orig_denoise_step = model.denoise_step
    v_t_steps: list = []

    def denoise_step_capture(state, prefix_pad_masks, past_key_values, x_t, timestep):
        if "past_keys" not in captured:
            keys, values = Pi05TensorRTExecutor._stack_past_key_value_tensors(past_key_values)
            captured["past_keys"] = keys.clone()
            captured["past_values"] = values.clone()
        v_t = orig_denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)
        v_t_steps.append(v_t.clone())
        return v_t

    model.denoise_step = denoise_step_capture

    # ---- hook sample_actions：捕获最终 x_t（未经 output transform） ----
    orig_sample_actions = model.sample_actions
    final_actions: list = []

    def sample_actions_capture(device, observation, **kw):
        out = orig_sample_actions(device, observation, **kw)
        final_actions.append(out.clone())
        return out

    model.sample_actions = sample_actions_capture
    if hasattr(policy, "_sample_actions"):
        policy._sample_actions = sample_actions_capture

    # ---- 固定 noise，跑一次 infer ----
    rng = np.random.default_rng(args.seed)
    horizon = int(model.config.action_horizon)
    action_dim = int(model.config.action_dim)
    noise = rng.standard_normal((horizon, action_dim)).astype(np.float32)

    example = build_example(args)
    result = policy.infer(example, noise=noise)

    assert "prefix_embs" in captured, "embed_prefix hook 未触发"
    assert "past_keys" in captured, "denoise_step hook 未触发（KV 未捕获）"
    assert len(v_t_steps) == args.num_steps, (
        f"v_t 步数 {len(v_t_steps)} != --num-steps {args.num_steps}"
    )
    assert final_actions, "sample_actions hook 未触发"

    # ---- 落盘 ----
    def save(name: str, arr) -> None:
        np.save(out_dir / f"{name}.npy", arr)
        print(f"  {name}.npy  shape={tuple(arr.shape)} dtype={arr.dtype}")

    print(f"dumping to {out_dir}:")
    for i, (img, mask) in enumerate(
        zip(captured["images"], captured["img_masks"], strict=True)
    ):
        save(f"image_{i}", to_npy(img).astype(np.float32))
        save(f"image_mask_{i}", to_npy(mask).astype(bool))
    save("lang_tokens", to_npy(captured["lang_tokens"]).astype(np.int64))
    save("lang_masks", to_npy(captured["lang_masks"]).astype(bool))
    save("noise", noise[None, ...])  # [1, H, D]

    save("prefix_embs", to_npy(captured["prefix_embs"]))
    save("prefix_pad_masks", to_npy(captured["prefix_pad_masks"]).astype(bool))
    save("past_keys", to_npy(captured["past_keys"]))
    save("past_values", to_npy(captured["past_values"]))
    for k, v_t in enumerate(v_t_steps):
        save(f"v_t_step{k}", to_npy(v_t))
    save("actions", to_npy(final_actions[0]))

    # ---- 原始观测 + 端到端 golden（pi05_infer 用） ----
    save("base_image", np.asarray(example["observation/image"], dtype=np.uint8))
    save("wrist_image", np.asarray(example["observation/wrist_image"], dtype=np.uint8))
    save("state", np.asarray(example["observation/state"], dtype=np.float32))
    prompt = example.get("prompt", args.prompt)
    if not isinstance(prompt, str):
        prompt = str(np.asarray(prompt).item())
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"  prompt.txt  {prompt!r}")
    save("actions_final", np.asarray(result["actions"], dtype=np.float32))

    # ---- AdaRMS 预计算模式：dump 每步 adarms_mod ----
    if args.adarms_precompute:
        from model_optimizer.infer.pi05_adarms import AdaRmsModulator

        modulator = AdaRmsModulator(model)
        device = captured["prefix_embs"].device
        bsize = captured["prefix_embs"].shape[0]
        for k in range(args.num_steps):
            t = 1.0 - k / args.num_steps
            timestep = torch.full((bsize,), t, dtype=torch.float32, device=device)
            save(f"adarms_mod_step{k}", to_npy(modulator(timestep)))

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
