"""compare 模式下 ViT get_image_features PT vs TRT tap。"""

from __future__ import annotations

from typing import Any

from .bundle_common import BundleProgress, policy_torch_model_for_perf


def install_vit_pt_trt_compare_taps(
    policy: Any,
    policy_trt: Any,
    progress: BundleProgress,
) -> None:
    def _install_vit_tap(pol: Any, tag: str) -> None:
        mdl = policy_torch_model_for_perf(pol)
        if mdl is None:
            raise RuntimeError(f"vit_pt_trt_compare: cannot resolve policy torch model for {tag}")
        mm = mdl.paligemma_with_expert.paligemma.model
        orig = getattr(mm, "get_image_features", None)
        if not callable(orig):
            raise RuntimeError(f"vit_pt_trt_compare: {tag} missing callable get_image_features")

        def _tensor_stats(x: Any) -> dict[str, Any] | None:
            try:
                import torch

                if not torch.is_tensor(x):
                    return None
                t = x.detach()
                tf = t.to(torch.float32)
                return {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "min": float(tf.amin().item()),
                    "max": float(tf.amax().item()),
                    "mean": float(tf.mean().item()),
                    "std": float(tf.std(unbiased=False).item()),
                }
            except Exception:
                return None

        def wrapped(pixel_values, *a, **kw):
            out = orig(pixel_values, *a, **kw)
            stats = _tensor_stats(pixel_values)
            try:
                out_det = out.detach()
            except Exception:
                out_det = out
            try:
                lst = getattr(mm, f"_webui_vit_calls_{tag}", None)
                if not isinstance(lst, list):
                    lst = []
                lst.append({"in_stats": stats, "out": out_det})
                setattr(mm, f"_webui_vit_calls_{tag}", lst)
            except Exception:
                setattr(mm, f"_webui_last_vit_in_stats_{tag}", stats)
                setattr(mm, f"_webui_last_vit_out_{tag}", out_det)
            return out

        mm.get_image_features = wrapped

        def fetch_and_clear():
            calls = getattr(mm, f"_webui_vit_calls_{tag}", None)
            last_out = getattr(mm, f"_webui_last_vit_out_{tag}", None)
            last_stats = getattr(mm, f"_webui_last_vit_in_stats_{tag}", None)
            for nm in (
                f"_webui_vit_calls_{tag}",
                f"_webui_last_vit_out_{tag}",
                f"_webui_last_vit_in_stats_{tag}",
            ):
                if hasattr(mm, nm):
                    try:
                        delattr(mm, nm)
                    except Exception:
                        pass
            if isinstance(calls, list) and calls:
                return calls
            return [{"in_stats": last_stats, "out": last_out}]

        setattr(pol, f"_webui_fetch_vit_io_{tag}", fetch_and_clear)

    progress.emit("vit_compare", "compare_mode：安装 ViT get_image_features tap（PT vs TRT）…")
    _install_vit_tap(policy, "pt")
    _install_vit_tap(policy_trt, "trt")
    progress.emit("vit_compare", "ViT tap 已安装（将按 chunk 推送 PT↔TRT 摘要）")
