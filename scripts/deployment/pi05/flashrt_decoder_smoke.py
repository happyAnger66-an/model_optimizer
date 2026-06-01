"""仓内 FlashRT decoder 冒烟/数值对比工具（Pi0.5 Thor）。

两种模式：

1) ``--mode repack``（默认，**无需 kernel/数据/GPU 即可跑**）
   - 加载 pi05 权重 → 跑权重 repack + AdaRMS 预计算；
   - 断言 ``dec_{qkv,o,gu,d}_flat`` / ``_ae_w_scales`` / ``sa/sf/fs`` / ``rope`` 的元素数与
     ``pipeline.decoder_forward`` 的 GEMM 步进一致（这是移植最大的风险点）；
   - 打印权重 scale 统计。用于上 Thor 前的离线自检。

2) ``--mode compare``（**Thor 上需先构建 flash_rt_kernels.so**）
   - 用一条 libero 示例，先跑原始 pytorch ``sample_actions`` 拿参考 raw action chunk；
   - 复用相同 observation + noise，跑仓内 FlashRT decoder 整循环；
   - 报告逐 action 的 max abs diff / 相对误差 / cosine，验证 §8.2 的 KV/RoPE 数值一致性。

用法：
  python scripts/deployment/pi05/flashrt_decoder_smoke.py \
    --model-path /srcs/openpi/pytorch_pi05_libero/ --config pi05_libero --mode repack

  # Thor 上数值对比（先 build kernel，见 build_flashrt_kernels.sh）
  python scripts/deployment/pi05/flashrt_decoder_smoke.py \
    --model-path /srcs/openpi/pytorch_pi05_libero/ --config pi05_libero \
    --mode compare --build-dir /workspace/flash_rt/build
"""

from __future__ import annotations

import argparse
import logging
import sys
import types

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("flashrt_smoke")


# ─────────────────────────────────────────────────────────────────────
#  模型加载
# ─────────────────────────────────────────────────────────────────────
def _load_model(model_path: str, config_ref: str):
    from openpi.policies import policy_config
    from openpi.training import config as _config

    from model_optimizer.models.pi05.model_pi05 import Pi05Model

    cfg = _config.get_config(config_ref)
    policy = policy_config.create_trained_policy(cfg, model_path)
    model = Pi05Model(policy).model
    return policy, model


def _model_dims(model) -> dict[str, int]:
    layers = model.paligemma_with_expert.gemma_expert.model.layers
    return {
        "Da": int(model.action_in_proj.out_features),
        "Ha": int(layers[0].mlp.gate_proj.out_features),
        "num_layers": len(layers),
        "Sa": int(model.config.action_horizon),
        "action_dim": int(model.action_in_proj.in_features),
        "NH": 8,
        "HD": 256,
    }


# ─────────────────────────────────────────────────────────────────────
#  Mode: repack（无 kernel）
# ─────────────────────────────────────────────────────────────────────
def run_repack(model, *, use_fp8: bool, device: str, enc_seq: int) -> int:
    from model_optimizer.infer.native.flashrt_decoder import (
        precompute_adarms_styles,
        build_dec_rope,
        repack_decoder_weights,
        state_dict_getter,
    )

    d = _model_dims(model)
    Da, Ha, L, Sa, NH, HD = d["Da"], d["Ha"], d["num_layers"], d["Sa"], d["NH"], d["HD"]
    logger.info("model dims: %s", d)

    sd = dict(model.state_dict())
    getter = state_dict_getter(sd)

    rep = repack_decoder_weights(
        getter, num_layers=L, num_q_heads=NH, steps=10, use_fp8=use_fp8, device=device
    )

    # 期望元素数（与 pipeline.decoder_forward 的 K/N 步进核对）
    exp = {
        "dec_qkv_flat": L * Da * 2560,
        "dec_o_flat": L * (NH * HD) * Da,
        "dec_gu_flat": L * Da * (Ha * 2),
        "dec_d_flat": L * Ha * Da,
        "ae_w_scales": L * 4,
    }
    ok = 0
    fail = 0
    for name, want in exp.items():
        got = int(getattr(rep, name).reshape(-1).numel())
        status = "OK" if got == want else "MISMATCH"
        if got != want:
            fail += 1
        else:
            ok += 1
        logger.info("[repack] %-14s numel=%d expect=%d -> %s", name, got, want, status)

    # 单例
    logger.info(
        "[repack] singletons: ain_w%s ain_b%s aow%s aob%s",
        tuple(rep.ain_w.shape), tuple(rep.ain_b.shape),
        tuple(rep.aow.shape), tuple(rep.aob.shape),
    )
    if use_fp8:
        s = rep.ae_w_scales
        logger.info(
            "[repack] ae_w_scales: min=%.3e max=%.3e mean=%.3e",
            float(s.min()), float(s.max()), float(s.mean()),
        )

    # AdaRMS 预计算 + rope
    styles = precompute_adarms_styles(getter, Sa=Sa, Da=Da, num_layers=L, steps=10, device=device)
    rope = build_dec_rope(enc_seq, Sa, device=device, head_dim=HD)
    exp_style = {
        "sa_all": (10 * L * Sa, 3 * Da),
        "sf_all": (10 * L * Sa, 3 * Da),
        "fs_all": (10 * Sa, 3 * Da),
    }
    for name, want_shape in exp_style.items():
        got_shape = tuple(getattr(styles, name).shape)
        status = "OK" if got_shape == want_shape else "MISMATCH"
        if got_shape != want_shape:
            fail += 1
        else:
            ok += 1
        logger.info("[precompute] %-7s shape=%s expect=%s -> %s", name, got_shape, want_shape, status)
    rope_ok = tuple(rope.shape) == (Sa, HD)
    logger.info("[precompute] rope shape=%s expect=%s -> %s", tuple(rope.shape), (Sa, HD), "OK" if rope_ok else "MISMATCH")
    fail += 0 if rope_ok else 1
    ok += 1 if rope_ok else 0

    logger.info("=== repack smoke: %d ok, %d fail ===", ok, fail)
    return 0 if fail == 0 else 1


# ─────────────────────────────────────────────────────────────────────
#  Mode: compare（Thor，需 kernel）
# ─────────────────────────────────────────────────────────────────────
def _capture_reference(policy, model, device: str):
    """跑一次原始 sample_actions，捕获 (observation, noise) 与参考 raw action chunk。"""
    from openpi.policies.libero_policy import make_libero_example

    orig = model.sample_actions
    cap: dict = {}

    def _cap(self_m, dev, observation, noise=None, num_steps=10):
        if noise is None:
            bsize = int(observation.state.shape[0])
            noise = self_m.sample_noise(
                (bsize, int(self_m.config.action_horizon), int(model.action_in_proj.in_features)),
                dev,
            )
        cap["device"] = dev
        cap["observation"] = observation
        cap["noise"] = noise
        cap["ref"] = orig(dev, observation, noise=noise, num_steps=num_steps)
        return cap["ref"]

    model.sample_actions = types.MethodType(_cap, model)
    try:
        if hasattr(policy, "_sample_actions"):
            policy._sample_actions = model.sample_actions
        _ = policy.infer(make_libero_example())
    finally:
        model.sample_actions = orig
        if hasattr(policy, "_sample_actions"):
            policy._sample_actions = orig
    return cap


def run_compare(policy, model, *, build_dir, fmha_so, use_fp8, device, act_scales_path, calibrate) -> int:
    from model_optimizer.infer.native.pi05_executor import Pi05NativeExecutor

    cap = _capture_reference(policy, model, device)
    ref = cap["ref"].detach().to(torch.float32)
    logger.info("captured ref action chunk: shape=%s", tuple(ref.shape))

    # 安装仓内 FlashRT decoder（sample_actions 整循环替换）
    import addict

    ex = Pi05NativeExecutor(policy, torch.bfloat16)
    ex.load_model(addict.Dict({
        "enable_expert": False,
        "enable_denoise": True,
        "perf": False,
        "flashrt_decoder": True,
        "flashrt_build_dir": build_dir or "",
        "flashrt_fmha_so": fmha_so or "",
        "flashrt_use_fp8": use_fp8,
        "flashrt_act_scales_path": act_scales_path or "",
        "flashrt_calibrate": calibrate,
    }))

    flash = model.sample_actions(cap["device"], cap["observation"], noise=cap["noise"])
    flash = flash.detach().to(torch.float32)

    if ex._flashrt_backend is None:
        logger.error(
            "FlashRT backend 未构建（很可能 kernel .so 未找到 → 已回退原始路径）。"
            "请确认 --build-dir 指向 Thor 上的 flash_rt_kernels.so。"
        )
        return 2

    diff = (flash - ref).abs()
    denom = ref.abs().clamp_min(1e-6)
    rel = (diff / denom)
    cos = torch.nn.functional.cosine_similarity(
        flash.reshape(-1), ref.reshape(-1), dim=0
    )
    logger.info("=== compare (flashrt vs pytorch raw sample_actions) ===")
    logger.info("max_abs_diff = %.4e", float(diff.max()))
    logger.info("mean_abs_diff= %.4e", float(diff.mean()))
    logger.info("max_rel_diff = %.4e", float(rel.max()))
    logger.info("cosine_sim   = %.6f", float(cos))
    passed = float(diff.max()) < 5e-2 and float(cos) > 0.99
    logger.info("=== compare verdict: %s ===", "PASS" if passed else "CHECK (见 §8.2 KV/RoPE 一致性)")
    return 0 if passed else 1


def main() -> int:
    p = argparse.ArgumentParser(description="仓内 FlashRT decoder 冒烟/数值对比")
    p.add_argument("--model-path", required=True, help="pi05 checkpoint 目录")
    p.add_argument("--config", required=True, help="openpi config 名（如 pi05_libero）")
    p.add_argument("--mode", choices=["repack", "compare"], default="repack")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use-fp8", action="store_true", default=True)
    p.add_argument("--no-fp8", dest="use_fp8", action="store_false")
    p.add_argument("--enc-seq", type=int, default=818, help="repack 模式下构 rope 用的 prefix 长度")
    p.add_argument("--build-dir", default="", help="flash_rt_kernels*.so 所在目录（compare 模式）")
    p.add_argument("--fmha-so", default="", help="可选 libfmha_fp16_strided.so 路径")
    p.add_argument("--act-scales-path", default="", help="act scales JSON（离线量化）")
    p.add_argument("--calibrate", action="store_true", help="compare 模式首跑导出 act scales")
    args = p.parse_args()

    logger.info("loading model: %s (config=%s)", args.model_path, args.config)
    policy, model = _load_model(args.model_path, args.config)

    if args.mode == "repack":
        return run_repack(model, use_fp8=args.use_fp8, device=args.device, enc_seq=args.enc_seq)
    return run_compare(
        policy, model,
        build_dir=args.build_dir, fmha_so=args.fmha_so,
        use_fp8=args.use_fp8, device=args.device,
        act_scales_path=args.act_scales_path, calibrate=args.calibrate,
    )


if __name__ == "__main__":
    sys.exit(main())
