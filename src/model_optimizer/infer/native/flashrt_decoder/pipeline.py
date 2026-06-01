"""Pi0.5 Thor decoder 推理/校准（移植自 FlashRT ``models/pi05/pipeline_thor.py``）。

逐字移植 ``decoder_forward`` / ``_decoder_forward_fp16`` / ``decoder_forward_calibrate``，
仅做两处适配：
  1) ``fvk`` 由调用方注入（:mod:`.kernels` 按 .so 路径加载，不 import flash_rt 包）；
  2) GPU 指针小工具改为本包 :mod:`.cuda_helpers`。

算子语义/调用顺序与 FlashRT 完全一致（18 层 × 10 步，静态 FP8）。
"""

from __future__ import annotations

import math

from .cuda_helpers import gpu_copy, gpu_sync, gpu_zero, measure_scale_gpu


# ══════════════════════════════════════════════════════════════════
# Decoder (18 layers, 10 diffusion steps, static FP8)
# ══════════════════════════════════════════════════════════════════

def decoder_forward(ctx, fvk, bufs, weights, dims, stream=0, *, attn=None, use_fp8=True):
    """Full AE decoder forward pass ≡ pi05 ae_forward_static（静态 FP8）。

    bufs/weights/dims 的指针约定见 FlashRT pipeline_thor.decoder_forward 文档串。
    """
    if not use_fp8:
        return _decoder_forward_fp16(ctx, fvk, bufs, weights, dims, stream, attn=attn)
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
        # ── Action input: noise → x ──
        fvk.gmm_fp16(ctx, noise, ain_w, x, S, D, 32, 0.0, stream)
        fvk.add_bias_fp16(x, ain_b, S, D, stream)

        for l in range(layers):
            si = (s * layers + l) * S * D3
            sa_ptr = sa + si * 2
            sf_ptr = sf + si * 2

            # ── C1: Fused AdaRMSNorm → FP8 with static scale ──
            act_scale_qkv = act_scales + (l * 4 + 0) * 4
            fvk.fused_adarms_fp8_static_fp16(x, sa_ptr, xn_fp8, gate, S, D, act_scale_qkv, stream)

            # ── C2: QKV GEMM with descale ──
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
            fvk.fp8_gemm_descale_fp16(ctx_fp8, ow_ptr, fg, S, D, NH * HD,
                                      act_scale_o, w_scale_o, stream)

            # ── C4→C5: gate×residual + AdaRMSNorm → FP8 ──
            act_scale_gu = act_scales + (l * 4 + 2) * 4
            fvk.gate_res_adarms_fp8_static_fp16(fg, gate, x, sf_ptr,
                                                xn_fp8, gate, S, D, act_scale_gu, stream)

            # ── C5: Gate+Up merged GEMM ──
            w_scale_gu = w_scales + (l * 4 + 2) * 4
            gw_ptr = gw + l * D * H * 2
            fvk.fp8_gemm_descale_fp16(xn_fp8, gw_ptr, fg, S, H * 2, D,
                                      act_scale_gu, w_scale_gu, stream)

            # ── C6: SiLU(gate) × up → FP8 ──
            act_scale_down = act_scales + (l * 4 + 3) * 4
            fvk.gate_geglu_merged_fp8_fp16(fg, hid_fp8, S, H, act_scale_down, stream)

            # ── C6: Down GEMM ──
            w_scale_down = w_scales + (l * 4 + 3) * 4
            dw_ptr = dw + l * H * D
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

def _decoder_forward_fp16(ctx, fvk, bufs, weights, dims, stream=0, *, attn=None):
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

def decoder_forward_calibrate(ctx, fvk_mod, bufs, weights, dims, calib_scales_ptr, stream=0):
    """校准 decoder FP8 激活 scale（离线量化能力）。

    每个量化点两遍：1) FP16 kernel → GPU 端测 amax；2) 用该 scale 跑 FP8 kernel。
    结果写入 ``calib_scales_ptr``（layers*4 float32）。
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

    gpu_copy(calib_scales_ptr, calib_buf, layers * 4 * 4, stream)
    gpu_sync(stream)
