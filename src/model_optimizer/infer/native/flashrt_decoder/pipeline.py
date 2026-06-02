"""Pi0.5 Thor decoder 推理/校准（移植自 FlashRT ``models/pi05/pipeline_thor.py``）。

逐字移植 ``decoder_forward`` / ``_decoder_forward_fp16`` / ``decoder_forward_calibrate``，
仅做两处适配：
  1) ``fvk`` 由调用方注入（:mod:`.kernels` 按 .so 路径加载，不 import flash_rt 包）；
  2) GPU 指针小工具改为本包 :mod:`.cuda_helpers`。

算子语义/调用顺序与 FlashRT 完全一致（18 层 × 10 步，静态 FP8）。
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING

from .cuda_helpers import gpu_copy, gpu_sync, gpu_zero, measure_scale_gpu

if TYPE_CHECKING:
    from model_optimizer.infer.perf import StagePerfCollector


# ══════════════════════════════════════════════════════════════════
# Decoder (18 layers, 10 diffusion steps, static FP8)
# ══════════════════════════════════════════════════════════════════

def decoder_forward(
    ctx,
    fvk,
    bufs,
    weights,
    dims,
    stream=0,
    *,
    attn=None,
    use_fp8=True,
    perf: StagePerfCollector | None = None,
):
    """Full AE decoder forward pass ≡ pi05 ae_forward_static（静态 FP8）。

    bufs/weights/dims 的指针约定见 FlashRT pipeline_thor.decoder_forward 文档串。
    """
    if not use_fp8:
        return _decoder_forward_fp16(ctx, fvk, bufs, weights, dims, stream, attn=attn, perf=perf)
    S = dims['S']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    steps = dims['steps']
    layers = dims['layers']
    enc_seq = dims['enc_seq']
    total_keys = dims['total_keys']
    D3 = 3 * D
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    noise = bufs['noise']
    x = bufs['x']
    xn = bufs['xn']
    gate = bufs['gate']
    qkv = bufs['qkv']
    logits = bufs['logits']
    attn_out = bufs['attn_out']
    hid = bufs['hid']
    fg = bufs['fg']
    xn_fp8 = bufs['xn_fp8']
    hid_fp8 = bufs['hid_fp8']
    ctx_fp8 = bufs['ctx_fp8']

    ain_w = weights['ain_w']
    ain_b = weights['ain_b']
    sa = weights['sa']
    qw = weights['qw']
    Kc = weights['Kc']
    Vc = weights['Vc']
    ow = weights['ow']
    sf = weights['sf']
    gw = weights['gw']
    dw = weights['dw']
    aow = weights['aow']
    aob = weights['aob']
    fs = weights['fs']
    rope = weights['rope']
    w_scales = weights['w_scales']
    act_scales = weights['act_scales']

    for s in range(steps):
        # 外层循环 ≡ openpi ``PI0Pytorch.sample_actions`` 里 ``while time >= -dt/2`` 的一次迭代，
        # 即单次 ``denoise_step(...)`` 调用（flow matching 的一个扩散步）。
        step_ctx = perf.timed(f"denoise.step.{s}") if perf is not None else nullcontext()
        with step_ctx:
            # ── Action input projection（≡ openpi ``embed_suffix`` 里的 ``action_in_proj``）──
            #
            # OpenPI 对应代码（``pi0_pytorch.py``）::
            #
            #   def denoise_step(..., x_t, timestep):
            #       suffix_embs, ..., adarms_cond = self.embed_suffix(state, x_t, timestep)
            #       ...
            #
            #   def embed_suffix(..., noisy_actions, timestep):
            #       action_emb = self.action_in_proj(noisy_actions)   # ← 本段 kernel 等价于此
            #       # pi05 另有时序分支 time_mlp → adarms_cond，在 FlashRT 里不在这里算，
            #       # 而是预计算进权重侧 ``sa/sf/fs``，由下面 C1 的 fused_adarms 消费。
            #
            # 张量对应关系（单 batch，S = Sa = action_horizon token 数）::
            #
            #   bufs['noise']  ↔  ``x_t`` / ``noisy_actions``，形状 ``[S, 32]``
            #                     （32 = ``config.action_dim``，flow matching 当前动作噪声状态）
            #   bufs['x']      ↔  ``action_emb`` 进入 expert 前的隐状态，形状 ``[S, D]``
            #                     （D = action expert 隐藏维，``action_in_proj.out_features``）
            #   weights['ain_w'] / ['ain_b']  ↔  ``self.action_in_proj.weight/bias``
            #
            # ``gmm_fp16(ctx, A, B, C, M, N, K, alpha, stream)``::
            #   计算 ``C[M,N] = alpha * C + A[M,K] @ B[K,N]``（此处 alpha=0 → 覆盖写 C）。
            #   这里 M=S, K=32, N=D： ``x = noise @ ain_w``，把动作空间 32 维抬升到 decoder 宽度 D。
            fvk.gmm_fp16(ctx, noise, ain_w, x, S, D, 32, 0.0, stream)
            # ``add_bias_fp16``:: ``x += ain_b``，与 ``nn.Linear`` 的 bias 项一致。
            fvk.add_bias_fp16(x, ain_b, S, D, stream)
            # 此后 ``x`` 作为第 0 层 transformer 的残差流输入，进入下面 ``for l in range(layers)``
            # 的 C1…C7（≡ ``denoise_step`` 里 ``paligemma_with_expert.forward`` + 各层 FFN/Attn）。

            for l in range(layers):
                si = (s * layers + l) * S * D3
                sa_ptr = sa + si * 2
                sf_ptr = sf + si * 2

                # ── C1: Fused AdaRMSNorm → FP8 with static scale ──
                #
                # OpenPI 对应（pi05，``denoise_step`` → ``paligemma_with_expert.forward``）::
                #
                #   # ``embed_suffix`` 已算好 ``adarms_cond``（≡ ``time_mlp(timestep)``，形状 ``[B, Da]``）::
                #   time_emb = silu(silu(time_mlp_in(sinusoidal_emb)) @ time_mlp_out)   # pi0_pytorch.py
                #   adarms_cond = time_emb
                #
                #   # Expert 第 ``l`` 层 self-attn 前（``gemma_pytorch.py``，仅 ``i=1`` 分支）::
                #   hidden_states, gate = layer.input_layernorm(
                #       hidden_states, cond=adarms_cond[1])
                #
                #   # ``GemmaRMSNorm.forward``（``modeling_gemma.py``）::
                #   modulation = self.dense(cond)          # Linear(Da → 3*D)
                #   scale, shift, gate = chunk(modulation, 3)
                #   x_norm = rmsnorm(x) * (1 + scale) + shift
                #   # 随后 ``q_proj/k_proj/v_proj(x_norm)`` → 本管线 C2 的 QKV GEMM
                #
                # FlashRT 把 ``dense(cond)`` 预计算进 ``sa``（``precompute.py::precompute_adarms_styles``）::
                #   sa[s,l,:] = time_emb @ input_layernorm.dense.weight.T + bias   # 每 token 一行，长度 3*D
                # ``sa_ptr`` 指向当前步 ``s``、层 ``l`` 的那段调制向量；kernel 内做 RMSNorm + scale/shift，
                # 再按静态标定尺度量化到 ``xn_fp8``，并写出 ``gate`` 供 C4 的 ``gate_res_adarms`` 做 gated residual。
                # 下游 C2 ``fp8_gemm_descale_fp16`` 见 ``docs/optimizer/flashrt/fp8_gemm_descale_fp16.md``。
                #
                # ``act_scale_qkv``：本层 QKV 路径的 FP8 激活标定（``l*4+0`` 槽位），OpenPI 无对应项。
                act_scale_qkv = act_scales + (l * 4 + 0) * 4
                fvk.fused_adarms_fp8_static_fp16(x, sa_ptr, xn_fp8, gate, S, D, act_scale_qkv, stream)

                # ── C2: QKV GEMM with descale（详见 docs/optimizer/flashrt/fp8_gemm_descale_fp16.md）──
                #
                # ``xn_fp8 × qw_fp8 → qkv(fp16)``：Tensor Core FP8 GEMM，epilogue 乘 ``s_act * s_w``，
                # **直接**写 fp16 ``qkv``，不是「先输出 fp8 再反量化」。
                # ``act_scale_qkv`` 须与 C1 相同（C1 量化激活、C2 descale 共用）。
                #
                # OpenPI 等价：``q_proj/k_proj/v_proj(input_layernorm 输出)`` 三次 Linear → 此处合并为
                # 一次 ``qw`` ``[D,2560]`` GEMM（``weights.py`` repack）。
                w_scale_qkv = w_scales + (l * 4 + 0) * 4
                qw_ptr = qw + l * D * 2560
                fvk.fp8_gemm_descale_fp16(xn_fp8, qw_ptr, qkv, S, 2560, D,
                                          act_scale_qkv, w_scale_qkv, stream)

                # ── C2b: Fused RoPE + QKV split + KV cache ──
                kv_offset = l * total_keys * HD + enc_seq * HD
                fvk.qkv_split_rope_kvcache_fp16(qkv, rope, attn_out, Kc, Vc,
                                                S, Q_dim, K_dim, HD, 2560,
                                                kv_offset, HD, stream)

                # ── C3: Cross-attention ──
                if attn is not None:
                    attn.run("decoder", l, q_seq=S, kv_seq=total_keys, stream=stream)
                else:
                    K_ptr = Kc + l * total_keys * HD * 2
                    V_ptr = Vc + l * total_keys * HD * 2
                    fvk.attention_qkv_fp16(ctx, attn_out, K_ptr, V_ptr,
                                           logits, attn_out,
                                           S, total_keys, NH, HD, attn_scale, stream)

                # ── C4: O proj ──
                act_scale_o = act_scales + (l * 4 + 1) * 4
                w_scale_o = w_scales + (l * 4 + 1) * 4
                fvk.quantize_fp8_static_fp16(attn_out, ctx_fp8, act_scale_o, S * NH * HD, stream)
                ow_ptr = ow + l * NH * HD * D
                # O-proj FP8 GEMM → fg(fp16)；``fp8_gemm_descale_fp16.md`` §5 槽位 k=1
                fvk.fp8_gemm_descale_fp16(ctx_fp8, ow_ptr, fg, S, D, NH * HD,
                                          act_scale_o, w_scale_o, stream)

                # ── C4→C5: gate×residual + AdaRMSNorm → FP8 ──
                act_scale_gu = act_scales + (l * 4 + 2) * 4
                fvk.gate_res_adarms_fp8_static_fp16(fg, gate, x, sf_ptr,
                                                    xn_fp8, gate, S, D, act_scale_gu, stream)

                # ── C5: Gate+Up merged GEMM ──
                w_scale_gu = w_scales + (l * 4 + 2) * 4
                gw_ptr = gw + l * D * H * 2
                # gate+up 合并 GEMM → fg(fp16)；``fp8_gemm_descale_fp16.md`` §5 槽位 k=2
                fvk.fp8_gemm_descale_fp16(xn_fp8, gw_ptr, fg, S, H * 2, D,
                                          act_scale_gu, w_scale_gu, stream)

                # ── C6: SiLU(gate) × up → FP8 ──
                act_scale_down = act_scales + (l * 4 + 3) * 4
                fvk.gate_geglu_merged_fp8_fp16(fg, hid_fp8, S, H, act_scale_down, stream)

                # ── C6: Down GEMM ──
                w_scale_down = w_scales + (l * 4 + 3) * 4
                dw_ptr = dw + l * H * D
                # down GEMM → fg(fp16)；``fp8_gemm_descale_fp16.md`` §5 槽位 k=3
                fvk.fp8_gemm_descale_fp16(hid_fp8, dw_ptr, fg, S, D, H,
                                          act_scale_down, w_scale_down, stream)

                # ── C7→C1_next: gate×residual + next AdaRMSNorm → FP8 ──
                if l < layers - 1:
                    si_next = (s * layers + l + 1) * S * D3
                    sa_next_ptr = sa + si_next * 2
                    act_scale_next = act_scales + ((l + 1) * 4 + 0) * 4
                    fvk.gate_res_adarms_fp8_static_fp16(fg, gate, x, sa_next_ptr,
                                                        xn_fp8, gate, S, D, act_scale_next, stream)
                else:
                    fvk.gate_res_fp16(fg, gate, x, S * D, stream)

            # ── Final: AdaRMSNorm + action output ──
            fi = s * S * D3
            fs_ptr = fs + fi * 2
            fvk.adarms_fp16(x, fs_ptr, xn, gate, S, D, stream)

            fvk.gmm_fp16(ctx, xn, aow, noise, S, 32, D, 1.0, stream)
            fvk.add_bias_fp16(noise, aob, S, 32, stream)


# ══════════════════════════════════════════════════════════════════
# FP16 decoder path (no quantization, FP16 weights, baseline only)
# ══════════════════════════════════════════════════════════════════

def _decoder_forward_fp16(
    ctx,
    fvk,
    bufs,
    weights,
    dims,
    stream=0,
    *,
    attn=None,
    perf: StagePerfCollector | None = None,
):
    """FP16-only decoder forward（结构镜像 FP8 路径，每个 GEMM 用 ``gmm_fp16``）。"""
    S = dims['S']; D = dims['D']; H = dims['H']
    NH = dims['NH']; HD = dims['HD']
    steps = dims['steps']; layers = dims['layers']
    enc_seq = dims['enc_seq']; total_keys = dims['total_keys']
    D3 = 3 * D
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    noise = bufs['noise']; x = bufs['x']; xn = bufs['xn']
    gate = bufs['gate']; qkv = bufs['qkv']; logits = bufs['logits']
    attn_out = bufs['attn_out']; fg = bufs['fg']; hid = bufs['hid']

    ain_w = weights['ain_w']; ain_b = weights['ain_b']
    sa = weights['sa']; qw = weights['qw']
    Kc = weights['Kc']; Vc = weights['Vc']
    ow = weights['ow']; sf = weights['sf']
    gw = weights['gw']; dw = weights['dw']
    aow = weights['aow']; aob = weights['aob']
    fs = weights['fs']; rope = weights['rope']

    for s in range(steps):
        step_ctx = perf.timed(f"denoise.step.{s}") if perf is not None else nullcontext()
        with step_ctx:
            fvk.gmm_fp16(ctx, noise, ain_w, x, S, D, 32, 0.0, stream)
            fvk.add_bias_fp16(x, ain_b, S, D, stream)

            for l in range(layers):
                si = (s * layers + l) * S * D3
                sa_ptr = sa + si * 2
                sf_ptr = sf + si * 2

                # C1: AdaRMSNorm (FP16, no FP8 quantize)
                fvk.adarms_fp16(x, sa_ptr, xn, gate, S, D, stream)

                # C2: QKV GEMM (FP16 NN; weight is [K, 2560])
                qw_ptr = qw + l * D * 2560 * 2  # FP16 = 2 bytes/elem
                fvk.gmm_fp16(ctx, xn, qw_ptr, qkv, S, 2560, D, 0.0, stream)

                # C2b: Split + RoPE + KV cache write
                kv_offset = l * total_keys * HD + enc_seq * HD
                fvk.qkv_split_rope_kvcache_fp16(qkv, rope, attn_out, Kc, Vc,
                                                S, Q_dim, K_dim, HD, 2560,
                                                kv_offset, HD, stream)

                # C3: Attention
                if attn is not None:
                    attn.run("decoder", l, q_seq=S, kv_seq=total_keys, stream=stream)
                else:
                    K_ptr = Kc + l * total_keys * HD * 2
                    V_ptr = Vc + l * total_keys * HD * 2
                    fvk.attention_qkv_fp16(ctx, attn_out, K_ptr, V_ptr,
                                           logits, attn_out,
                                           S, total_keys, NH, HD, attn_scale, stream)

                # C4: O proj
                ow_ptr = ow + l * NH * HD * D * 2  # FP16
                fvk.gmm_fp16(ctx, attn_out, ow_ptr, fg, S, D, NH * HD, 0.0, stream)

                # C4→C5: gated residual + post-attn AdaRMSNorm
                fvk.gate_res_fp16(fg, gate, x, S * D, stream)
                fvk.adarms_fp16(x, sf_ptr, xn, gate, S, D, stream)

                # C5: Gate+Up merged GEMM (weight is [K, 2H])
                gw_ptr = gw + l * D * H * 2 * 2  # FP16 = 2 bytes/elem; weight has 2H cols
                fvk.gmm_fp16(ctx, xn, gw_ptr, fg, S, H * 2, D, 0.0, stream)

                # C6: GELU(gate) × up (FP16)
                fvk.gate_geglu_merged_fp16(fg, hid, S, H, stream)

                # C6: Down GEMM (weight is [H, D])
                dw_ptr = dw + l * H * D * 2  # FP16
                fvk.gmm_fp16(ctx, hid, dw_ptr, fg, S, D, H, 0.0, stream)

                # C7: residual into x (next layer's C1 will do AdaRMS)
                fvk.gate_res_fp16(fg, gate, x, S * D, stream)

            # Final: AdaRMSNorm + action output
            fi = s * S * D3
            fs_ptr = fs + fi * 2
            fvk.adarms_fp16(x, fs_ptr, xn, gate, S, D, stream)
            fvk.gmm_fp16(ctx, xn, aow, noise, S, 32, D, 1.0, stream)
            fvk.add_bias_fp16(noise, aob, S, 32, stream)


# ══════════════════════════════════════════════════════════════════
# Calibration（离线量化：纯指针，框架无关）
# ══════════════════════════════════════════════════════════════════

def decoder_forward_calibrate(
    ctx, fvk_mod, bufs, weights, dims, calib_scales_ptr, stream=0, *, per_step_scales_ptr=0
):
    """校准 decoder FP8 激活 scale（离线量化能力）。

    每个量化点两遍：1) FP16 kernel → GPU 端测 amax；2) 用该 scale 跑 FP8 kernel。
    结果写入 ``calib_scales_ptr``（layers*4 float32，**最后一步 step 的 scale**，与 FlashRT 原版一致）。

    跨步 max-merge（改进项，不改 in-loop FP8 forward）：传 ``per_step_scales_ptr``
    （``steps*layers*4`` float32 设备 buffer）时，每个扩散步结束把当步 ``calib_buf`` 落盘到
    ``per_step_scales_ptr + s*layers*4``，由调用方在 driver/torch 侧对 step 轴取 max，
    得到"整 10 步最大激活"的 scale（FP8 不易饱和）。
    """
    S = dims['S']; D = dims['D']; H = dims['H']
    NH = dims['NH']; HD = dims['HD']
    steps = dims['steps']; layers = dims['layers']
    enc_seq = dims['enc_seq']; total_keys = dims['total_keys']
    Q_dim = NH * HD
    attn_scale = 1.0 / math.sqrt(float(HD))
    D3 = 3 * D

    noise = bufs['noise']; x = bufs['x']; xn = bufs['xn']
    gate_buf = bufs['gate']; qkv = bufs['qkv']; logits = bufs['logits']
    attn_out = bufs['attn_out']; hid = bufs['hid']; fg = bufs['fg']
    xn_fp8 = bufs['xn_fp8']; hid_fp8 = bufs['hid_fp8']; ctx_fp8 = bufs['ctx_fp8']

    ain_w = weights['ain_w']; ain_b = weights['ain_b']
    sa = weights['sa']; qw = weights['qw']
    Kc = weights['Kc']; Vc = weights['Vc']
    ow = weights['ow']; sf = weights['sf']
    gw = weights['gw']; dw = weights['dw']
    aow = weights['aow']; aob = weights['aob']
    fs = weights['fs']; rope = weights['rope']
    w_scales = weights['w_scales']

    calib_buf = bufs['calib_buf']          # layers*4 float32
    d_scale = bufs['d_scale']              # 1 float32
    hidden_scratch = bufs['hidden_scratch']  # S*H fp16
    fp8_scratch = bufs['fp8_scratch']      # S*max(D,H) fp8
    gpu_zero(calib_buf, layers * 4 * 4, stream)

    for s in range(steps):
        fvk_mod.gmm_fp16(ctx, noise, ain_w, x, S, D, 32, 0.0, stream)
        fvk_mod.add_bias_fp16(x, ain_b, S, D, stream)

        for l in range(layers):
            si = (s * layers + l) * S * D3
            sa_ptr = sa + si * 2
            sf_ptr = sf + si * 2

            # C1: AdaRMSNorm FP16 → measure amax → FP8
            fvk_mod.adarms_fp16(x, sa_ptr, xn, gate_buf, S, D, stream)
            measure_scale_gpu(fvk_mod, xn, S * D, d_scale, fp8_scratch, stream)
            gpu_sync(stream)
            cs_qkv = calib_buf + (l * 4 + 0) * 4
            gpu_copy(cs_qkv, d_scale, 4, stream)
            fvk_mod.fused_adarms_fp8_static_fp16(x, sa_ptr, xn_fp8, gate_buf,
                                                 S, D, cs_qkv, stream)

            # C2: QKV GEMM
            ws_qkv = w_scales + (l * 4 + 0) * 4
            qw_ptr = qw + l * D * 2560
            fvk_mod.fp8_gemm_descale_fp16(xn_fp8, qw_ptr, qkv, S, 2560, D,
                                          cs_qkv, ws_qkv, stream)

            # C2b: Split+RoPE
            kv_offset = l * total_keys * HD + enc_seq * HD
            fvk_mod.qkv_split_rope_kvcache_fp16(qkv, rope, attn_out, Kc, Vc,
                                                S, Q_dim, HD, HD, 2560,
                                                kv_offset, HD, stream)

            # C3: Attention
            K_ptr = Kc + l * total_keys * HD * 2
            V_ptr = Vc + l * total_keys * HD * 2
            fvk_mod.attention_qkv_fp16(ctx, attn_out, K_ptr, V_ptr,
                                       logits, attn_out,
                                       S, total_keys, NH, HD, attn_scale, stream)

            # C4: O proj — measure attn amax → FP8 → GEMM
            measure_scale_gpu(fvk_mod, attn_out, S * NH * HD, d_scale, fp8_scratch, stream)
            gpu_sync(stream)
            cs_o = calib_buf + (l * 4 + 1) * 4
            gpu_copy(cs_o, d_scale, 4, stream)
            ws_o = w_scales + (l * 4 + 1) * 4
            fvk_mod.quantize_fp8_static_fp16(attn_out, ctx_fp8, cs_o, S * NH * HD, stream)
            ow_ptr = ow + l * NH * HD * D
            fvk_mod.fp8_gemm_descale_fp16(ctx_fp8, ow_ptr, fg, S, D, NH * HD,
                                          cs_o, ws_o, stream)

            # C4→C5: gate×residual + AdaRMSNorm → measure → FP8
            fvk_mod.gate_res_fp16(fg, gate_buf, x, S * D, stream)
            fvk_mod.adarms_fp16(x, sf_ptr, xn, gate_buf, S, D, stream)
            measure_scale_gpu(fvk_mod, xn, S * D, d_scale, fp8_scratch, stream)
            gpu_sync(stream)
            cs_gu = calib_buf + (l * 4 + 2) * 4
            gpu_copy(cs_gu, d_scale, 4, stream)
            fvk_mod.quantize_fp8_static_fp16(xn, xn_fp8, cs_gu, S * D, stream)

            # C5: Gate+Up GEMM
            ws_gu = w_scales + (l * 4 + 2) * 4
            gw_ptr = gw + l * D * H * 2
            fvk_mod.fp8_gemm_descale_fp16(xn_fp8, gw_ptr, fg, S, H * 2, D,
                                          cs_gu, ws_gu, stream)

            # C6: GELU → measure → FP8
            fvk_mod.gate_geglu_merged_fp16(fg, hidden_scratch, S, H, stream)
            measure_scale_gpu(fvk_mod, hidden_scratch, S * H, d_scale, fp8_scratch, stream)
            gpu_sync(stream)
            cs_down = calib_buf + (l * 4 + 3) * 4
            gpu_copy(cs_down, d_scale, 4, stream)
            fvk_mod.gate_geglu_merged_fp8_fp16(fg, hid_fp8, S, H, cs_down, stream)

            # C6: Down GEMM
            ws_down = w_scales + (l * 4 + 3) * 4
            dw_ptr = dw + l * H * D
            fvk_mod.fp8_gemm_descale_fp16(hid_fp8, dw_ptr, fg, S, D, H,
                                          cs_down, ws_down, stream)

            # C7: gate×residual + next layer prep
            if l < layers - 1:
                si_next = (s * layers + l + 1) * S * D3
                sa_next_ptr = sa + si_next * 2
                fvk_mod.gate_res_fp16(fg, gate_buf, x, S * D, stream)
                fvk_mod.adarms_fp16(x, sa_next_ptr, xn, gate_buf, S, D, stream)
                measure_scale_gpu(fvk_mod, xn, S * D, d_scale, fp8_scratch, stream)
                gpu_sync(stream)
                cs_next = calib_buf + ((l + 1) * 4 + 0) * 4
                gpu_copy(cs_next, d_scale, 4, stream)
                fvk_mod.quantize_fp8_static_fp16(xn, xn_fp8, cs_next, S * D, stream)
            else:
                fvk_mod.gate_res_fp16(fg, gate_buf, x, S * D, stream)

        fi = s * S * D3
        fs_ptr = fs + fi * 2
        fvk_mod.adarms_fp16(x, fs_ptr, xn, gate_buf, S, D, stream)
        fvk_mod.gmm_fp16(ctx, xn, aow, noise, S, 32, D, 1.0, stream)
        fvk_mod.add_bias_fp16(noise, aob, S, 32, stream)

        # 跨步 max-merge（改进项）：落盘当步 layers*4 scale，供 driver 端对 step 轴取 max。
        if per_step_scales_ptr:
            gpu_copy(per_step_scales_ptr + s * layers * 4 * 4, calib_buf, layers * 4 * 4, stream)

    gpu_copy(calib_scales_ptr, calib_buf, layers * 4 * 4, stream)
    gpu_sync(stream)
