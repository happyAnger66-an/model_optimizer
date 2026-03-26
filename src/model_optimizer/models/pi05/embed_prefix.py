"""π₀.₅ 前缀嵌入子图：对齐 openpi ``PI0Pytorch.embed_prefix``（SigLIP 图像 token + 语言 embedding 拼接）。"""

from __future__ import annotations

import logging
import math
import os
import time
import torch
import torch.nn as nn
from termcolor import colored

from ..model import Model
from model_optimizer.quantization.quantization_utils import quantize_model
from model_optimizer.utils.utils import is_nvfp4_quantized, set_dynamic_quant

logger = logging.getLogger(__name__)


class Pi05EmbedPrefix(nn.Module, Model):
    """
    将多路相机图像经 SigLIP + projector 得到 patch embedding，与经 embedding 层
    缩放后的语言 token 嵌入在序列维拼接；并构造与 openpi 一致的 ``pad_masks`` /
    ``att_masks``（图像块、语言段均为 prefix 双向注意，``att_masks`` 全 0）。

    ONNX / TensorRT 导出时使用 ``forward(*inputs)``：按
    ``image_0, image_mask_0, …, lang_tokens, lang_masks`` 顺序传入，共
    ``2 * num_image_views + 2`` 个张量。
    """

    def __init__(
        self,
        vision_tower: nn.Module,
        multi_modal_projector: nn.Module,
        embed_tokens: nn.Module,
        *,
        hidden_size: int,
        num_image_views: int,
        max_token_len: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector
        self.embed_tokens = embed_tokens
        self.hidden_size = hidden_size
        self.num_image_views = num_image_views
        self.max_token_len = max_token_len
        self.device = next(self.vision_tower.parameters()).device

    @property
    def model(self):
        """与 Vit / LLM 一致，供量化后 NVFP4 等路径使用。"""
        return self.vision_tower

    def val(self, val_data, batch_size, output_dir):
        raise NotImplementedError(
            "Pi05EmbedPrefix.val 未实现：需提供 prefix 嵌入校准/对比数据与指标。"
        )

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """与 openpi ``PI0Pytorch.embed_prefix`` 相同签名与语义。"""
        if len(images) != self.num_image_views or len(img_masks) != self.num_image_views:
            raise ValueError(
                f"期望 {self.num_image_views} 路图像与 mask，"
                f"得到 images={len(images)}, img_masks={len(img_masks)}"
            )

        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        att_masks: list[int] = []

        for img, img_mask in zip(images, img_masks, strict=True):
            image_outputs = self.vision_tower(img)
            selected_image_feature = image_outputs.last_hidden_state
            img_emb = self.multi_modal_projector(selected_image_feature)
            img_emb = img_emb / math.sqrt(float(self.hidden_size))
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.embed_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(float(lang_emb_dim))

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        out_embs = torch.cat(embs, dim=1)
        out_pad = torch.cat(pad_masks, dim=1)
        att_t = torch.tensor(att_masks, dtype=torch.bool, device=out_pad.device)

        bsize = out_pad.shape[0]
        att_t = att_t[None, :].expand(bsize, att_t.shape[0])
        return out_embs, out_pad, att_t

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = 2 * self.num_image_views + 2
        if len(inputs) != n:
            raise ValueError(
                f"Pi05EmbedPrefix.forward 需要 {n} 个输入 "
                f"(每路 image + mask，再加 lang_tokens、lang_masks)，得到 {len(inputs)}"
            )
        images = [inputs[2 * i] for i in range(self.num_image_views)]
        img_masks = [inputs[2 * i + 1] for i in range(self.num_image_views)]
        lang_tokens = inputs[-2]
        lang_masks = inputs[-1]
        return self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def export(
        self,
        export_dir: str,
        export_dtype: torch.dtype = torch.bfloat16,
        dynamo: bool = False,
    ):
        """导出 ``embed_prefix.onnx``；输入输出与 ``forward`` 一致。"""
        self.eval().cuda()

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        max_token_len = self.max_token_len

        dummy: list[torch.Tensor] = []
        input_names: list[str] = []
        for i in range(self.num_image_views):
            dummy.append(
                torch.randn((1, 3, 224, 224), dtype=export_dtype, device="cuda"),
            )
            dummy.append(torch.ones((1,), dtype=torch.bool, device="cuda"))
            input_names.extend([f"image_{i}", f"image_mask_{i}"])

        dummy.append(
            torch.zeros((1, max_token_len), dtype=torch.int64, device="cuda"),
        )
        dummy.append(torch.ones((1, max_token_len), dtype=torch.bool, device="cuda"))
        input_names.extend(["lang_tokens", "lang_masks"])

        output_path = f"{output_dir}/embed_prefix.onnx"
        fallback_order = [dynamo]
        if not dynamo:
            fallback_order.append(True)
        else:
            fallback_order.append(False)

        def _export_kwargs(use_dynamo: bool) -> dict:
            if use_dynamo:
                from torch.export import Dim

                batch_dim = Dim("batch", min=1, max=4096)
                token_dim = Dim("token_len", min=1, max=max_token_len)
                ds_list: list[dict] = []
                for _ in range(self.num_image_views):
                    ds_list.append({0: batch_dim})  # image_i
                    ds_list.append({0: batch_dim})  # image_mask_i
                ds_list.append({0: batch_dim, 1: token_dim})  # lang_tokens
                ds_list.append({0: batch_dim, 1: token_dim})  # lang_masks
                # forward(*inputs) 在 torch.export 侧只有一个顶层参数名 "inputs"（可变参数元组）
                return {"dynamic_shapes": {"inputs": tuple(ds_list)}}

            dynamic_axes = {}
            for i in range(self.num_image_views):
                dynamic_axes[f"image_{i}"] = {0: "batch_size"}
                dynamic_axes[f"image_mask_{i}"] = {0: "batch_size"}
            dynamic_axes["lang_tokens"] = {0: "batch_size", 1: "token_len"}
            dynamic_axes["lang_masks"] = {0: "batch_size", 1: "token_len"}
            dynamic_axes["prefix_embs"] = {0: "batch_size", 1: "prefix_seq"}
            dynamic_axes["prefix_pad_masks"] = {0: "batch_size", 1: "prefix_seq"}
            dynamic_axes["prefix_att_masks"] = {0: "batch_size", 1: "prefix_seq"}
            return {"dynamic_axes": dynamic_axes}

        logger.info("Start export embed_prefix onnx ...")
        print(colored("Start Pi05 embed_prefix (Pi05EmbedPrefix) export onnx...", "green"))

        with torch.inference_mode():
            last_err = None
            used_dynamo = None
            for idx, use_dynamo in enumerate(fallback_order):
                export_kw = _export_kwargs(use_dynamo)
                try:
                    if idx > 0:
                        logger.warning(
                            "embed_prefix export fallback attempt %s: dynamo=%s",
                            idx,
                            use_dynamo,
                        )
                    torch.onnx.export(
                        self,
                        tuple(dummy),
                        output_path,
                        export_params=True,
                        input_names=input_names,
                        output_names=["prefix_embs", "prefix_pad_masks", "prefix_att_masks"],
                        opset_version=19,
                        dynamo=use_dynamo,
                        do_constant_folding=True,
                        **export_kw,
                    )
                    used_dynamo = use_dynamo
                    break
                except Exception as err:
                    last_err = err
                    logger.exception(
                        "embed_prefix export failed with dynamo=%s, will %s",
                        use_dynamo,
                        "fallback" if idx < len(fallback_order) - 1 else "raise",
                    )
            if used_dynamo is None:
                # If all attempts fail, re-raise the last one.
                raise RuntimeError("embed_prefix export failed with both dynamo and torchscript paths") from last_err

        end = time.time()
        logger.info(
            "export embed_prefix onnx to %s done cost:%ss (dynamo=%s)",
            output_dir,
            end - start,
            used_dynamo,
        )
        print(
            colored(
                f"Pi05 embed_prefix export onnx done to {output_path} cost:{end - start}s",
                "green",
            )
        )
        return self

    def quantize(self, quant_cfg, calib_data, export_dir):
        calib_dataloader = self.get_calibrate_dataset(calib_data)
        quantize_model(self, quant_cfg, calib_dataloader)
        self.is_quantized = True
        set_dynamic_quant(self, "fp16")

        self.export(export_dir, dynamo=False)
        onnx_path = f"{export_dir}/embed_prefix.onnx"
        if is_nvfp4_quantized(quant_cfg):
            print(colored("nvfp4 quantization detected, post processing...", "green"))
            self._nvfp4_post_processing(onnx_path, export_dir)

    @classmethod
    def construct_model(cls, pi05_model, dtype: torch.dtype | None = None):
        """从 ``Pi05Model`` 或 ``PI0Pytorch``（``policy._model``）构建，共享 ``paligemma_with_expert`` 权重。"""
        if not getattr(pi05_model.config, "pi05", False):
            raise ValueError("Pi05EmbedPrefix 仅支持 config.pi05 is True 的 PI0Pytorch 模型")

        obs_spec, _ = pi05_model.config.inputs_spec(batch_size=1)
        num_views = len(obs_spec.images)

        if dtype is not None:
            logger.debug("construct_model dtype=%s ignored (权重保持加载时 dtype)", dtype)

        paligemma = pi05_model.paligemma_with_expert.paligemma
        return cls(
            vision_tower=paligemma.model.vision_tower,
            multi_modal_projector=paligemma.model.multi_modal_projector,
            embed_tokens=paligemma.language_model.embed_tokens,
            hidden_size=int(paligemma.config.text_config.hidden_size),
            num_image_views=num_views,
            max_token_len=int(pi05_model.config.max_token_len),
        )

    @classmethod
    def construct_from_name_path(cls, model_name: str, model_path: str):
        from .model_pi05 import Pi05Model

        wrapper = Pi05Model.construct_from_name_path(model_name, model_path)
        return cls.construct_model(wrapper, dtype=torch.bfloat16)
