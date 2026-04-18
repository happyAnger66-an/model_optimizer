"""π₀.₅ 前缀嵌入子图：对齐 openpi ``PI0Pytorch.embed_prefix``（SigLIP 图像 token + 语言 embedding 拼接）。"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import time
import torch
import torch.nn as nn
from termcolor import colored
from tqdm import tqdm

from ..model import Model
from model_optimizer.calibrate.pi05_calib_load import open_pi05_calib_for_quantize
from model_optimizer.evaluate.metrics.pi05 import Pi05Metric
from model_optimizer.quantization.quantization_utils import quantize_model
from model_optimizer.utils.utils import is_nvfp4_quantized, set_dynamic_quant
from modelopt.torch.quantization.utils import export_torch_mode

logger = logging.getLogger(__name__)


def _patch_modelopt_quantizers_trt_hp_for_onnx(model: nn.Module) -> None:
    """与 ``whole_export._patch_modelopt_quantizers_trt_high_precision_for_onnx`` 一致，避免 FP8 ONNX 符号
    在 TRT strongly-typed 路径上与激活 dtype 不一致。"""
    override = os.environ.get("PI05_WHOLE_TRT_HP_DTYPE", "").strip()
    if override:
        hp = override
    else:
        try:
            p = next(model.parameters())
            if p.dtype == torch.float16:
                hp = "Half"
            elif p.dtype == torch.bfloat16:
                hp = "BFloat16"
            else:
                hp = "Float"
        except StopIteration:
            hp = "Float"
    n = 0
    for mod in model.modules():
        for qn in ("input_quantizer", "output_quantizer", "weight_quantizer"):
            q = getattr(mod, qn, None)
            if q is None:
                continue
            if hasattr(q, "_trt_high_precision_dtype"):
                setattr(q, "_trt_high_precision_dtype", hp)
                n += 1
    if n:
        logger.info("ModelOpt ONNX FP8: set _trt_high_precision_dtype=%r on %s quantizer(s)", hp, n)


@contextlib.contextmanager
def _siglip_vision_quantconv_disable_input_quant_for_torchscript_onnx(
    vision_tower: nn.Module | None,
) -> None:
    """缓解 TorchScript ONNX：``TRT_FP8DequantizeLinear`` 输出作为 ``Conv2d`` 输入时
    ``SymbolicValueError: convolution for kernel of unknown shape``。

    PyTorch ONNX Conv 符号化无法处理该 TRT 自定义 op 到 Conv 的数据流；导出窗口内对
    ``vision_tower`` 下各 ``QuantConv2d`` 临时 ``input_quantizer.disable()``，结束后恢复。
    权重侧 ``weight_quantizer`` 仍参与前向，图内仍可保留权重量化相关节点。

    调试若需关闭本缓解：环境变量 ``PI05_EMBED_PREFIX_ONNX_NO_VISION_IQ_OFF=1``。
    """
    if vision_tower is None:
        yield
        return
    if os.environ.get("PI05_EMBED_PREFIX_ONNX_NO_VISION_IQ_OFF", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    ):
        yield
        return
    try:
        from modelopt.torch.quantization.nn.modules.quant_conv import QuantConv2d
    except ImportError:
        yield
        return

    reenable: list = []
    try:
        for m in vision_tower.modules():
            if isinstance(m, QuantConv2d):
                iq = getattr(m, "input_quantizer", None)
                if iq is None or getattr(iq, "_disabled", False):
                    continue
                iq.disable()
                reenable.append(iq)
        if reenable:
            logger.info(
                "embed_prefix TorchScript ONNX: disabled input_quantizer on %s vision QuantConv2d "
                "(workaround for TRT_FP8DequantizeLinear -> Conv export)",
                len(reenable),
            )
        yield
    finally:
        for iq in reenable:
            iq.enable()


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

    def get_calibrate_dataset(self, calib_data):
        return open_pi05_calib_for_quantize(calib_data, component="pi05_embed_prefix")

    def val(self, val_data, batch_size, output_dir):
        val_datas = self.get_calibrate_dataset(val_data)

        def val_loop(model: torch.nn.Module, output_datas: list) -> None:
            try:
                n = len(val_datas)
            except TypeError:
                n = None
            if n is not None:
                print(f"Val embed_prefix on {n} samples...")
            else:
                print("Val embed_prefix (streaming calib data, total unknown)...")
            pbar = tqdm(val_datas, total=n, desc="Val embed_prefix", unit="num_samples")
            for data in pbar:
                if not isinstance(data, dict):
                    raise TypeError("embed_prefix calib 样本应为 dict（含 image_* / lang_*）。")
                batch = {k: v.to(model.device) for k, v in data.items()}
                out = model(**batch)
                output_datas.append(
                    {
                        "prefix_embs": out[0].to(torch.float32).detach().cpu().numpy(),
                        "prefix_pad_masks": out[1].to(torch.float32).detach().cpu().numpy(),
                        "prefix_att_masks": out[2].to(torch.float32).detach().cpu().numpy(),
                    }
                )

        if getattr(self, "is_quantized", False):
            print(colored("Quantized embed_prefix val", "green"))
            self.val_datas_after = []
            val_loop(self, self.val_datas_after)
            return Pi05Metric(self.val_datas_after)
        print(colored("Original embed_prefix val", "green"))
        self.val_datas_before = []
        val_loop(self, self.val_datas_before)
        return Pi05Metric(self.val_datas_before)

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
        # Keep external interface unchanged (image_0/image_1/...), but batch all views
        # into a single tower forward to reduce per-view launch overhead.
        batched_images = torch.stack(images, dim=1)  # [B, V, 3, H, W]
        bsize, num_views = batched_images.shape[:2]
        flat_images = batched_images.reshape(
            bsize * num_views,
            batched_images.shape[2],
            batched_images.shape[3],
            batched_images.shape[4],
        )

        image_outputs = self.vision_tower(flat_images)
        selected_image_feature = image_outputs.last_hidden_state
        flat_img_emb = self.multi_modal_projector(selected_image_feature)
        flat_img_emb = flat_img_emb / math.sqrt(float(self.hidden_size))

        num_img_embs = flat_img_emb.shape[1]
        img_emb = flat_img_emb.reshape(bsize, num_views, num_img_embs, flat_img_emb.shape[-1])
        img_emb = img_emb.reshape(bsize, num_views * num_img_embs, flat_img_emb.shape[-1])

        stacked_img_masks = torch.stack(img_masks, dim=1)  # [B, V]
        img_pad_masks = stacked_img_masks[:, :, None].expand(bsize, num_views, num_img_embs)
        img_pad_masks = img_pad_masks.reshape(bsize, num_views * num_img_embs)

        embs.append(img_emb)
        pad_masks.append(img_pad_masks)

        lang_emb = self.embed_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(float(lang_emb_dim))

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        out_embs = torch.cat(embs, dim=1)
        out_pad = torch.cat(pad_masks, dim=1)
        bsize = out_pad.shape[0]
        seq_len = out_embs.shape[1]
        # 与原先 ``[0]*len -> bool`` 等价（全 False），避免 ``torch.tensor(Python list)`` 在 ONNX 追踪下不稳定
        att_t = torch.zeros((1, seq_len), device=out_embs.device, dtype=torch.bool).expand(bsize, -1)
        return out_embs, out_pad, att_t

    def forward(
        self,
        *inputs: torch.Tensor,
        **batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ONNX 导出：位置参数 ``*inputs``；ModelOpt 量化：``model(**dict)`` 与校准分片字段一致。"""
        if batch:
            return self._forward_keyword_batch(batch)
        n = 2 * self.num_image_views + 2
        if len(inputs) != n:
            raise ValueError(
                f"Pi05EmbedPrefix.forward 需要 {n} 个位置参数张量 "
                f"(每路 image + mask，再加 lang_tokens、lang_masks)，得到 {len(inputs)}；"
                f"或使用关键字参数 image_0, image_mask_0, …, lang_tokens, lang_masks。"
            )
        images = [inputs[2 * i] for i in range(self.num_image_views)]
        img_masks = [inputs[2 * i + 1] for i in range(self.num_image_views)]
        lang_tokens = inputs[-2]
        lang_masks = inputs[-1]
        return self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def _forward_keyword_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.num_image_views
        missing = [f"image_{i}" for i in range(n) if f"image_{i}" not in batch]
        missing += [f"image_mask_{i}" for i in range(n) if f"image_mask_{i}" not in batch]
        for k in ("lang_tokens", "lang_masks"):
            if k not in batch:
                missing.append(k)
        if missing:
            raise KeyError(f"embed_prefix 量化 forward 缺少键: {missing}")
        images = [batch[f"image_{i}"] for i in range(n)]
        img_masks = [batch[f"image_mask_{i}"] for i in range(n)]
        return self.embed_prefix(images, img_masks, batch["lang_tokens"], batch["lang_masks"])

    def export(
        self,
        export_dir: str,
        export_dtype: torch.dtype = torch.bfloat16,
        dynamo: bool = False,
        mode = None
    ):
        """导出 ``embed_prefix.onnx``；输入输出与 ``forward`` 一致。

        若已 **ModelOpt 量化**（``is_quantized``），与 ``Expert.quantize`` → ``export(..., dynamo=False)``
        及 ``Expert.export_onnx`` 一致，**仅使用 ``dynamo=False``**（TorchScript ONNX 追踪）。
        ModelOpt 的 fake 量化在 CUDA 上前向会调用 ``torch.ops.tensorrt.quantize_op``；旧式 ONNX 导出
        对该路径走 ``autograd.Function.symbolic`` 落到 Q/DQ 等 ONNX 算子，而 **``dynamo=True``**
        走 ``torch.export`` 时图中仍保留 ``tensorrt.quantize_op``，新导出器无法翻译，会报
        ``No ONNX function found for tensorrt.quantize_op``。
        非量化路径仍可按参数尝试 ``dynamo=True``（与未量化 Expert 默认一致）。
        已量化时导出前会打 ModelOpt ``_trt_high_precision_dtype`` patch，并在 **整个** ONNX 导出外包
        ``export_torch_mode()``（与 ModelOpt 对 ``torch.export`` 的用法一致，亦有助于部分 TorchScript
        ONNX 路径下量化 Conv/Linear 的追踪稳定性）。

        **TorchScript ONNX + SigLIP ``QuantConv2d``**：若出现 ``TRT_FP8DequantizeLinear`` 接 ``Conv`` 的
        ``kernel of unknown shape``，在 ``dynamo=False`` 导出内会自动临时关闭 ``vision_tower`` 内各
        ``QuantConv2d`` 的 ``input_quantizer``（见 ``_siglip_vision_quantconv_disable_input_quant_for_torchscript_onnx``）。

        **Dynamo / torch.export**：语言序列维若声明 ``Dim("token_len", …)`` 与示例长度组合，易触发
        ``ConstraintViolationError``（图把 ``L`` specialize 成常数与可变约束矛盾）。此处对 **dynamo 路径**
        仅将 **batch（维 0）** 标为动态，**``lang_*`` 第 2 维固定为 ``max_token_len``**（与 Pi0.5 配置一致）；
        传统 ``dynamo=False`` 仍可用 ``dynamic_axes`` 标 ``token_len``（浮点导出）。
        """
        self.eval().cuda()

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        max_token_len = int(self.max_token_len)
        if max_token_len < 1:
            max_token_len = 1
        lang_seq_len = max_token_len

        dummy: list[torch.Tensor] = []
        input_names: list[str] = []
        for i in range(self.num_image_views):
            dummy.append(
                torch.randn((1, 3, 224, 224), dtype=export_dtype, device="cuda"),
            )
            dummy.append(torch.ones((1,), dtype=torch.bool, device="cuda"))
            input_names.extend([f"image_{i}", f"image_mask_{i}"])

        dummy.append(
            torch.zeros((1, lang_seq_len), dtype=torch.int64, device="cuda"),
        )
        dummy.append(torch.ones((1, lang_seq_len), dtype=torch.bool, device="cuda"))
        input_names.extend(["lang_tokens", "lang_masks"])

        output_path = f"{export_dir}/embed_prefix.onnx"
        quantized = bool(getattr(self, "is_quantized", False))
        if quantized:
            # 与 Expert 量化导出一致：只用 TorchScript ONNX；dynamo 无法降低 tensorrt.quantize_op。
            fallback_order = [False]
            if dynamo:
                logger.info(
                    "embed_prefix 已量化：忽略 dynamo=True，固定使用 dynamo=False（与 Expert.export 一致）。"
                )
        else:
            # 未量化：仅按用户选择；不自动 dynamo=True，避免无谓告警。
            fallback_order = [True, False] if dynamo else [False]

        def _export_kwargs(use_dynamo: bool) -> dict:
            if use_dynamo:
                from torch.export import Dim

                batch_dim = Dim("batch", min=1, max=4096)
                ds_list: list[dict] = []
                for _ in range(self.num_image_views):
                    ds_list.append({0: batch_dim})  # image_i
                    ds_list.append({0: batch_dim})  # image_mask_i
                # 仅 batch 动态；语言长度固定为 max_token_len，避免 token_len Dim 与 specialize 冲突
                ds_list.append({0: batch_dim})
                ds_list.append({0: batch_dim})
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

        if quantized:
            _patch_modelopt_quantizers_trt_hp_for_onnx(self)

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
                    export_ctx = (
                        export_torch_mode() if quantized else contextlib.nullcontext()
                    )
                    ts_iq_ctx = (
                        _siglip_vision_quantconv_disable_input_quant_for_torchscript_onnx(
                            self.vision_tower
                        )
                        if (quantized and not use_dynamo)
                        else contextlib.nullcontext()
                    )
                    with export_ctx:
                        with ts_iq_ctx:
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
                raise RuntimeError(
                    f"embed_prefix ONNX export failed after {len(fallback_order)} attempt(s): {last_err!r}"
                ) from last_err

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
        set_dynamic_quant(self, "bf16")

        self.export(export_dir, dynamo=False)  # 已量化时固定 TorchScript ONNX（与 Expert 一致）
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
