# Pi0.5 Thor：视觉 + 语言前缀如何拼接并送入 Encoder（Prefill）

本文总结 **FlashRT**（`flash_rt/frontends/torch/pi05_thor.py` + `hardware/thor/shared_primitives.py`）与 **OpenPI**（`openpi/models_pytorch/pi0_pytorch.py`）在 **「前缀 embedding → LLM / encoder prefill」** 上的对应关系。FlashRT 可视为 Pi0.5 的 **手搓 CUDA Graph + 内核** 实现。

---

## 1. 与 OpenPI 的对应关系

**OpenPI**：`embed_prefix` 产出 **`prefix_embs`**，再交给 PaliGemma **`forward(..., inputs_embeds=[prefix_embs, None], use_cache=True)`** 做 **encoder prefill 并写 KV**。

```386:400:/home/zhangxa/codes/model_optimizer/third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
```

**FlashRT Thor**：前缀的物理载体是 GPU 缓冲 **`_enc_x[:Se]`**（FP16）。由 **两段 CUDA Graph** 完成：

1. **`_siglip_graph`**：`patch` → **SigLIP** → **PostLN + 投影 + 语言 memcpy**，写满 **`_enc_x` 的前缀段**。
2. **`_enc_ae_graph`**：**`encoder_forward`** 读 **`_enc_x`**，写 **`Kc`/`Vc`**（prefill），再接 **decoder**。

---

## 2. `img_list`：观测 → `_img_buf` → SigLIP Graph

**`infer`**（非 CFG）中收集多视角图像，与校准路径逻辑一致：

```1936:1959:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/flash_rt/frontends/torch/pi05_thor.py
        if 'images' in observation:
            img_list = observation['images']
        else:
            img_list = [observation['image']]
            if nv >= 2:
                img_list.append(
                    observation.get('wrist_image', observation['image']))
            if nv >= 3:
                img_list.append(
                    observation.get('wrist_image_right', img_list[-1]))

        ...
        images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
        self._img_buf.upload(images_np)

        # ---- Graph 1: SigLIP + PostLN ----
        self._siglip_graph.replay()
```

- **`nv = num_views`**：只取 **`img_list[:nv]`**。
- **三视角**：第三路缺省 **`wrist_image_right`** 时用 **`img_list[-1]`** 填位（与校准循环同构）。
- **`_siglip_graph.replay()`** 内顺序（capture 时固定）：**`_patch_embed_ops` → `siglip_forward` → `_postln_project_ops`**。

---

## 3. `set_prompt`：语言 embedding、长度、`Se` 与 Graph 重建

**不负责上传当前帧图像**；负责 **token 化 / embedding**、**encoder 序列长度 `Se`**、**`total_keys`**、**`ThorFlashAttnBackend`**、**` _lang_emb` / `_S_lang`**（及 RoPE、时间条件、**重捕 `_siglip_graph` / `_enc_ae_graph`** 等）。

要点摘录：

```893:944:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/flash_rt/frontends/torch/pi05_thor.py
        Se = S_sig + prompt_len
        if Se % 2 != 0:
            Se += 1
        self.Se = Se
        self.total_keys = Se + self.Sa
        ...
        actual_lang = Se - S_sig
        if actual_lang > prompt_len:
            embeds = torch.cat([embeds, embeds[-1:]], dim=0)
        self._lang_emb = embeds
        self._S_lang = actual_lang
```

- **`S_sig`**：SigLIP 视觉 token 总数（如 `num_views * 256`）。
- **`Se`**：**视觉 + 语言** 槽位总长度；**偶数**约束服务于 **cuBLASLt FP8**。
- **`_lang_emb`**：GPU 上 **`[S_lang, D_enc]`**，供 **`postln_project`** 拷入 **`_enc_x`** 的语言段；**`S_lang`** 可与 **`prompt_len`** 不同（因 **`Se - S_sig`** pad）。

---

## 4. 拼接位置：`postln_project`（`_postln_project_ops`）

**`_postln_project_ops`** 调用 **`postln_project`**（`shared_primitives.py`）：

```175:189:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/flash_rt/hardware/thor/shared_primitives.py
    fvk.layer_norm_fp16(x_sig, weights['ln_w'], weights['ln_b'], scratch,
                        S_sig, D_sig, 1e-6, stream)
    gemm.fp16_nn(scratch, weights['proj_w'], enc_x, S_sig, D_enc, D_sig, stream)
    fvk.add_bias_fp16(enc_x, weights['proj_b'], S_sig, D_enc, stream)

    # enc_x[S_sig : S_sig+S_lang, :] = lang_emb
    nbytes = S_lang * D_enc * 2  # fp16
    dst = enc_x + S_sig * D_enc * 2
    _crt.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(weights['lang_emb']),
                          ctypes.c_size_t(nbytes), 3, ctypes.c_void_p(stream))
```

**`_enc_x` 前缀布局**：

| 行区间 | 含义 |
|--------|------|
| **`[0, S_sig)`** | PostLN + **`D_sig → D_enc`** 投影后的 **视觉 token**（输入为 **`_sig_x`**，即 SigLIP 最终输出）。 |
| **`[S_sig, S_sig+S_lang)`** | **`_lang_emb`**（**`set_prompt`** 写入，与 tokenizer + pad 对齐）。 |

有效前缀长度与 **`Se`**、`set_prompt` 里 **`self.Se`** 一致。

---

## 5. Encoder Prefill：`_enc_ae_graph` 中的 `encoder_forward`

**SigLIP graph** 结束后，**`_enc_x[:Se]`** 即为 **与 OpenPI `prefix_embs` 等价的前缀**（dtype / 归一化细节由各自 pipeline 定义）。

**Encoder** 在 **`_capture_enc_ae_graph`** 捕获的 graph 内执行；`enc_bufs['x']` 指向 **`_enc_x`**：

```1264:1277:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/flash_rt/frontends/torch/pi05_thor.py
        enc_bufs = {
            'x':       self._enc_x.data_ptr(),
            ...
        }
        ...
            encoder_forward(self._gemm, fvk, enc_bufs, enc_weights,
                            enc_dims, stream=s_int, attn=self._attn)
```

**`infer`** 中的调用顺序：

```1958:1975:/home/zhangxa/codes/model_optimizer/third_party/FlashRT/flash_rt/frontends/torch/pi05_thor.py
        self._siglip_graph.replay()
        ...
        self._enc_ae_graph.replay()
```

即：**先构造前缀（视觉 + 语言）→ 再 18 层 encoder + decoder**（decoder 侧噪声等在同 graph 内，与 OpenPI 分步 API 不同，但 **encoder 读入的前缀张量角色一致**）。

---

## 6. 对照表

| 环节 | OpenPI | FlashRT Thor Pi0.5 |
|------|--------|---------------------|
| 视觉 | `embed_prefix` 内 | `_img_buf` → patch → **`siglip_forward`** → **`_sig_x`** |
| 语言 | `lang_tokens` → embedding | **`set_prompt`** → **`_lang_emb`**（含 pad / `sqrt(D_enc)` 等） |
| 拼前缀 | 得到 **`prefix_embs`** | **`postln_project`**：vision → **`enc_x[0:S_sig]`**；**`cudaMemcpy`** **`_lang_emb` → `enc_x[S_sig:]`** |
| Encoder prefill | `paligemma...forward(..., use_cache=True)` | **`encoder_forward`**（**`_enc_ae_graph`**）读 **`_enc_x`**，写 **`Kc`/`Vc`** |

---

## 7. CFG / Batched 变体（仅索引）

- **CFG**：`_infer_cfg` 仍先组 **`img_list`**、**`upload`**，再由 **`Pi05ThorCFGPipeline`** 等 **多次 replay SigLIP** 并切换 **`_lang_emb`**（cond / uncond）；前缀写入 **`_enc_x` 或 ` _enc_x_b2`** 的细节见 `pi05_thor.py` 内 CFG 相关注释。
- **`infer_batch` / B=2**：SigLIP 仍多为 **B=1 graph 多次**，再 **`encoder_forward_b2`**；与上表同一前缀语义，仅缓冲按 batch 展开。

---

## 8. 一句话

**图像经 `img_list` → `_img_buf`，在 `_siglip_graph` 里变成 `_sig_x` 并经 PostLN+投影写入 `_enc_x` 前 `S_sig` 行；`set_prompt` 把文本变成 `_lang_emb` 并在同 graph 尾部拷入 `_enc_x` 后续行；`_enc_ae_graph` 里的 `encoder_forward` 以 `_enc_x` 为输入完成 encoder prefill（KV 写入 `Kc`/`Vc`）。** 这与 OpenPI **`embed_prefix` + `inputs_embeds` prefill** 同构。
