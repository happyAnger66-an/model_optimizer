import argparse
import ctypes
import logging
import os
import time
from collections.abc import Sequence
import fnmatch

import tensorrt as trt

from termcolor import colored
import onnx
from onnx import TensorProto
# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

print_color = "green"

# -----------------------------------------------------------------------------
# Precision/dtype preflight checks
# -----------------------------------------------------------------------------


def infer_onnx_float_precision(onnx_path: str) -> str:
    """粗略推断 ONNX 图中主要浮点精度（用于 build_cfg.precision 一致性检查）。"""
    model = onnx.load(onnx_path, load_external_data=False)
    float_types: set[int] = set()

    def _collect_from_value_info(v) -> None:
        try:
            t = v.type.tensor_type
            et = int(t.elem_type)
        except Exception:
            return
        if et in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16):
            float_types.add(et)

    for v in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        _collect_from_value_info(v)
    for init in model.graph.initializer:
        et = int(init.data_type)
        if et in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16):
            float_types.add(et)

    if len(float_types) == 0:
        return "unknown"
    if len(float_types) > 1:
        return "mixed"
    only = next(iter(float_types))
    if only == TensorProto.BFLOAT16:
        return "bf16"
    if only == TensorProto.FLOAT16:
        return "fp16"
    return "fp32"


def validate_precision_matches_onnx(onnx_path: str, precision: str) -> None:
    """若 build_cfg.precision 与 ONNX 浮点 dtype 明显不一致，则抛错，避免错误 kernel 导致数值塌缩。"""
    onnx_prec = infer_onnx_float_precision(onnx_path)
    if onnx_prec in ("unknown", "mixed"):
        logger.warning("ONNX float precision appears to be %s (precision=%s).", onnx_prec, precision)
        return

    tokens = set(normalize_precision_tokens(precision))

    def _has(t: str) -> bool:
        return t in tokens

    # If user requests explicit bf16 builder support, ONNX should not be fp16-only (and vice versa).
    if onnx_prec == "bf16" and _has("fp16") and not _has("bf16"):
        raise ValueError(
            f"Precision mismatch: ONNX graph uses {onnx_prec}, but build_cfg.precision tokens {sorted(tokens)} "
            "enable FP16 without BF16. Enable bf16 in precision (e.g., 'bf16' or 'bf16,fp16') or change ONNX dtype."
        )
    if onnx_prec == "fp16" and _has("bf16") and not _has("fp16"):
        raise ValueError(
            f"Precision mismatch: ONNX graph uses {onnx_prec}, but build_cfg.precision tokens {sorted(tokens)} "
            "enable BF16 without FP16. Enable fp16 in precision (e.g., 'fp16' or 'bf16,fp16') or change ONNX dtype."
        )


def normalize_precision_tokens(precision: str) -> list[str]:
    """Split ``precision`` into tokens.

    Supports comma-separated lists, e.g. ``\"bf16,fp16\"``.
    Tokens are stripped and lowercased; empty entries are ignored.
    """
    if precision is None:
        return []
    raw = str(precision).strip().lower()
    if not raw:
        return []
    parts = []
    for tok in raw.replace(";", ",").split(","):
        t = tok.strip().lower()
        if not t:
            continue
        parts.append(t)
    # de-dupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for t in parts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def apply_builder_precision_flags(config: trt.IBuilderConfig, precision_tokens: list[str]) -> None:
    """Apply TensorRT ``BuilderFlag`` precision flags.

    Notes:
        - ``fp32`` enables no precision flags (TRT default).
        - Duplicate tokens are already removed by :func:`normalize_precision_tokens`.
    """
    unknown: list[str] = []
    for tok in precision_tokens:
        if tok in ("fp32", "float32"):
            logger.info("precision token %r: FP32 baseline (no extra BuilderFlag)", tok)
            print(colored(f"precision token {tok}: FP32 baseline (no extra BuilderFlag)", print_color))
            continue
        if tok == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 mode (BuilderFlag.FP16)")
            print(colored("Enabled FP16 mode (BuilderFlag.FP16)", print_color))
            continue
        if tok == "bf16":
            config.set_flag(trt.BuilderFlag.BF16)
            logger.info("Enabled BF16 mode (BuilderFlag.BF16)")
            print(colored("Enabled BF16 mode (BuilderFlag.BF16)", print_color))
            continue
        if tok == "fp8":
            config.set_flag(trt.BuilderFlag.FP8)
            logger.info("Enabled FP8 mode (BuilderFlag.FP8)")
            print(colored("Enabled FP8 mode (BuilderFlag.FP8)", print_color))
            continue
        if tok in ("int4", "w4"):
            config.set_flag(trt.BuilderFlag.INT4)
            logger.info("Enabled INT4 mode (BuilderFlag.INT4)")
            print(colored("Enabled INT4 mode (BuilderFlag.INT4)", print_color))
            continue
        unknown.append(tok)

    if unknown:
        raise ValueError(
            f"Unknown precision token(s): {unknown}. "
            "Supported tokens: fp32, fp16, bf16, fp8, int4"
        )


def _trt_dtype_from_str(dtype: str) -> trt.DataType:
    t = str(dtype).strip().lower()
    if t in ("fp32", "float", "float32"):
        return trt.float32
    if t in ("fp16", "half", "float16"):
        return trt.float16
    if t in ("bf16", "bfloat16"):
        return trt.bfloat16
    if t in ("int32", "i32"):
        return trt.int32
    if t in ("int8", "i8"):
        return trt.int8
    if t in ("int64", "i64"):
        return trt.int64
    if t in ("bool",):
        return trt.bool
    raise ValueError(f"Unsupported TensorRT dtype string: {dtype!r}")


def apply_layer_precision_overrides(
    network: trt.INetworkDefinition,
    overrides: dict[str, str],
    *,
    match: str = "substring",
    set_output_types: bool = True,
) -> int:
    """Force per-layer precision (non-STRONGLY_TYPED networks only).

    Args:
        network: TRT network after ONNX parsing.
        overrides: Mapping from layer match key -> dtype string, e.g. {"/action_in_proj/MatMul": "fp32"}.
        match:
            - "substring": match when key is a substring of layer.name
            - "exact": match when key == layer.name
        set_output_types: If True, also set all outputs' dtype to the same dtype.

    Returns:
        Number of layers updated (may be 0 if no match).
    """
    if not overrides:
        return 0
    match_mode = str(match).strip().lower()
    if match_mode not in (
        "substring",
        "exact",
        "glob",
        "onnx_substring",
        "onnx_exact",
        "onnx_glob",
    ):
        raise ValueError(
            f"Unsupported match mode: {match!r} "
            "(expected 'substring', 'exact', 'glob', 'onnx_substring', 'onnx_exact', or 'onnx_glob')"
        )

    updated = 0
    updated_layers = []
    for li in range(network.num_layers):
        layer = network.get_layer(li)
        lname = str(getattr(layer, "name", "") or "")
        # TRT 10+ exposes parsed metadata that often contains strings like:
        # "[ONNX Layer: /action_in_proj/MatMul]"
        try:
            lmeta = str(getattr(layer, "metadata", "") or "")
        except Exception:
            lmeta = ""
        for key, dtype_str in overrides.items():
            k = str(key)
            if match_mode == "exact":
                ok = k == lname
            elif match_mode == "substring":
                ok = k in lname
            elif match_mode == "glob":
                ok = fnmatch.fnmatch(lname, k)
            elif match_mode == "onnx_exact":
                ok = k == lmeta
            elif match_mode == "onnx_substring":
                ok = k in lmeta
            else:  # onnx_glob
                ok = fnmatch.fnmatch(lmeta, k)
            if not ok:
                continue
            dt = _trt_dtype_from_str(dtype_str)
            # Safety: only apply precision overrides to float-like layers/tensors.
            # Avoid breaking Constant/Shape/etc. that produce INT/BOOL tensors.
            try:
                ltype = getattr(layer, "type", None)
            except Exception:
                ltype = None
            if ltype is not None and ltype == trt.LayerType.CONSTANT:
                logger.info("Skip precision override for Constant layer %r (metadata=%r)", lname, lmeta)
                print(colored(f"Skip precision override for Constant layer: {lname}", "yellow"))
                break

            float_types = {trt.float32, trt.float16, trt.bfloat16}
            bad_io = False
            # Check output tensor dtypes
            for oi in range(layer.num_outputs):
                try:
                    out_t = layer.get_output(oi)
                    if out_t is None:
                        continue
                    out_dt = getattr(out_t, "dtype", None)
                    if out_dt is not None and out_dt not in float_types:
                        bad_io = True
                        break
                except Exception:
                    # If we cannot introspect, don't block; let TRT validate later.
                    pass
            # Check input tensor dtypes
            if not bad_io:
                for ii in range(layer.num_inputs):
                    try:
                        in_t = layer.get_input(ii)
                        if in_t is None:
                            continue
                        in_dt = getattr(in_t, "dtype", None)
                        if in_dt is not None and in_dt not in float_types:
                            bad_io = True
                            break
                    except Exception:
                        pass
            if bad_io:
                logger.info(
                    "Skip precision override for non-float I/O layer %r (requested=%s, metadata=%r)",
                    lname, dtype_str, lmeta,
                )
                print(colored(f"Skip precision override for non-float I/O layer: {lname}", "yellow"))
                break

            try:
                layer.precision = dt
            except Exception as exc:
                logger.warning("Failed to set layer.precision for %r: %s", lname, exc)
                continue
            if set_output_types:
                for oi in range(layer.num_outputs):
                    try:
                        layer.set_output_type(oi, dt)
                    except Exception as exc:
                        logger.warning(
                            "Failed to set layer output type for %r out%d: %s",
                            lname, oi, exc,
                        )
            updated += 1
            updated_layers.append(lname)
            logger.info("Forced layer precision: %s -> %s", lname, dtype_str)
            print(colored(f"Forced layer precision: {lname} -> {dtype_str}", print_color))
            # Give users a moment to spot the override in logs.
            time.sleep(0.5)
            break
    if updated_layers:
        logger.info("Precision overrides applied to layers: %s", ", ".join(updated_layers))
        print(colored(f"Precision overrides applied to layers: {', '.join(updated_layers)}", print_color))
        time.sleep(3)
    return updated


def _load_trt_plugin_shared_libraries(
    paths: Sequence[str],
    *,
    logger_: logging.Logger,
) -> None:
    """在解析 ONNX / 构建网络之前加载自定义插件 .so（注册 IPluginCreator 等）。"""
    for p in paths:
        ap = os.path.abspath(os.path.expanduser(p))
        if not os.path.isfile(ap):
            raise FileNotFoundError(f"Plugin library not found: {ap}")
        logger_.info("Loading TensorRT plugin library: %s", ap)
        ctypes.CDLL(ap, mode=ctypes.RTLD_GLOBAL)


def _network_creation_flags(*, strongly_typed: bool) -> int:
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if strongly_typed:
        st_flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
        if st_flag is not None:
            flags |= 1 << int(st_flag)
    return flags


def build_engine(
    onnx_path: str,
    engine_path: str,
    use_cudagraph: bool = True,
    precision: str = "bf16",
    workspace_mb: int = 8192,
    min_shapes: dict = None,
    opt_shapes: dict = None,
    max_shapes: dict = None,
    *,
    plugin_lib_paths: Sequence[str] | None = None,
    init_builtin_trt_plugins: bool = True,
    strongly_typed_network: bool = False,
    layer_precision_overrides: dict[str, str] | None = None,
    layer_precision_match: str = "substring",
    set_layer_output_types: bool = True,
    precision_constraints: str = "prefer",
    debug_output_tensors: Sequence[str] | None = None,
    debug_dump_tensor_names: bool = False,
):
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode. Accepts either a single mode (``fp32``/``fp16``/``bf16``/``fp8``/``int4``)
            or a comma-separated list such as ``\"bf16,fp16\"`` to enable multiple ``BuilderFlag`` bits
            (**only meaningful when strongly_typed_network is False**).
        workspace_mb: Workspace size in MB
        min_shapes: Minimum input shapes (dict: name -> shape tuple)
        opt_shapes: Optimal input shapes (dict: name -> shape tuple)
        max_shapes: Maximum input shapes (dict: name -> shape tuple)
        plugin_lib_paths:
            自定义 TensorRT 插件 .so 的绝对或相对路径列表（如 TensorRT-Edge-LLM 编译出的
            ``lib*.so``）。TensorRT Python API 没有单独的 “插件搜索路径” 参数；必须在
            ``OnnxParser.parse_from_file`` **之前** 用 ``ctypes.CDLL(..., RTLD_GLOBAL)``
            加载，使插件向 TensorRT 注册。可与环境变量 ``LD_LIBRARY_PATH`` 配合（依赖的
            CUDA/TRT 库仍需能被动态链接器找到）。
        init_builtin_trt_plugins:
            若为 True，在加载自定义库之前调用 ``trt.init_libnvinfer_plugins``，注册随
            TensorRT 安装的 ``libnvinfer_plugin`` 等内置插件（常见 ONNX 自定义域会依赖）。
        strongly_typed_network:
            若为 True（默认），创建网络时附加 ``STRONGLY_TYPED``，与 TRT 10+ 在
            ``selectIODTypesForPluginV3...`` 中的类型推断一致；含 ``trt::AttentionPlugin``
            等自定义插件时，可避免出现
            ``doesn't report any supported format combinations``（弱类型图 + 插件 I/O
            不匹配）。若解析失败可改为 False 试旧行为。

    注意：TensorRT-Edge-LLM 的 AttentionPlugin 在 ONNX schema 中多为 **FP16** 张量；
    若 ``precision="bf16"`` 建引擎仍报插件格式组合错误，请将 **ONNX 导出为 FP16**
    （与插件一致）。

    **STRONGLY_TYPED 与 Builder 混精标志互斥**：开启 ``strongly_typed_network`` 时，
    不能再设置 ``FP16`` / ``BF16`` / ``FP8`` / ``INT4`` 等 ``BuilderFlag``；否则 TRT 报错
    ``!config.getFlag(BuilderFlag::kFP16)``。此时 ``precision`` 仅作日志提示，实际精度以
    ONNX 图中类型为准。
    """
    logger.info("=" * 80)
    logger.info("TensorRT Engine Builder")
    logger.info("=" * 80)
    logger.info(f"ONNX model: {onnx_path}")
    logger.info(f"Engine output: {engine_path}")
    logger.info(f"Precision: {precision.upper()}")
    logger.info(f"Workspace: {workspace_mb} MB")
    logger.info("=" * 80)

    print(colored("=" * 80, print_color))
    print(colored("TensorRT Engine Builder", print_color))
    print(colored("=" * 80), print_color)
    print(colored(f"ONNX model: {onnx_path}", print_color))
    print(colored(f"Engine output: {engine_path}", print_color))
    print(colored(f"Precision: {precision.upper()}", print_color))
    print(colored(f"Workspace: {workspace_mb} MB", print_color))
    print(colored("=" * 80, print_color))

    # Create TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    # 预检查：避免 ONNX dtype 与 build_cfg.precision 不一致导致 TRT kernel 选择错误、深层输出塌缩
    validate_precision_matches_onnx(onnx_path, precision)

    if init_builtin_trt_plugins:
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        logger.info("Initialized built-in TensorRT plugin registry (libnvinfer_plugin)")
        print(colored("Initialized built-in TensorRT plugin registry", print_color))

    if plugin_lib_paths:
        _load_trt_plugin_shared_libraries(plugin_lib_paths, logger_=logger)
        print(colored(f"Loaded {len(plugin_lib_paths)} custom plugin library(ies)", print_color))

    # Create builder and network
    logger.info("\n[Step 1/5] Creating TensorRT builder...")
    print(colored("\n[Step 1/5] Creating TensorRT builder...", print_color))
    builder = trt.Builder(TRT_LOGGER)
    net_flags = _network_creation_flags(strongly_typed=strongly_typed_network)
    network = builder.create_network(net_flags)
    if strongly_typed_network and getattr(
        trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None
    ) is not None:
        logger.info("Network flags: EXPLICIT_BATCH | STRONGLY_TYPED")
        print(colored("Network flags: EXPLICIT_BATCH | STRONGLY_TYPED", print_color))
    elif strongly_typed_network:
        logger.warning(
            "STRONGLY_TYPED not available in this TensorRT build; using EXPLICIT_BATCH only"
        )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    logger.info("\n[Step 2/5] Parsing ONNX model...")
    print(colored("\n[Step 2/5] Parsing ONNX model...", print_color))
    if not parser.parse_from_file(onnx_path):
        logger.error("Failed to parse ONNX file")
        for error in range(parser.num_errors):
            logger.error(parser.get_error(error))
        raise RuntimeError("ONNX parsing failed")

    # Parser successful. Network is loaded
    logger.info(f"Network inputs: {network.num_inputs}")
    print(colored(f"Network inputs: {network.num_inputs}", print_color))
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        logger.info(f"  Input {i}: {inp.name} {inp.shape}")
        print(colored(f"  Input {i}: {inp.name} {inp.shape}", print_color))

    logger.info(f"Network outputs: {network.num_outputs}")
    print(colored(f"Network outputs: {network.num_outputs}", print_color))
    for i in range(network.num_outputs):
        out = network.get_output(i)
        logger.info(f"  Output {i}: {out.name} {out.shape}")
        print(colored(f"  Output {i}: {out.name} {out.shape}", print_color))

    # Apply per-layer precision overrides (only for non-STRONGLY_TYPED networks)
    if layer_precision_overrides and not strongly_typed_network:
        n = apply_layer_precision_overrides(
            network,
            dict(layer_precision_overrides),
            match=layer_precision_match,
            set_output_types=bool(set_layer_output_types),
        )
        if n == 0:
            logger.warning(
                "layer_precision_overrides provided but no layer matched. "
                "Try a different key or set layer_precision_match to "
                "'exact' / 'glob' / 'onnx_substring' / 'onnx_exact' / 'onnx_glob'."
            )
            print(colored(
                "WARNING: layer_precision_overrides provided but no layer matched.",
                "yellow",
            ))

    def _iter_all_tensors():
        # Iterate tensors reachable from layers (best-effort; TRT Python API lacks a global tensor registry)
        for li in range(network.num_layers):
            layer = network.get_layer(li)
            # inputs
            for j in range(layer.num_inputs):
                t = layer.get_input(j)
                if t is not None:
                    yield t
            # outputs
            for j in range(layer.num_outputs):
                t = layer.get_output(j)
                if t is not None:
                    yield t

    def _find_tensor_by_name(name: str):
        target = str(name)
        for t in _iter_all_tensors():
            try:
                if t.name == target:
                    return t
            except Exception:
                continue
        return None

    if debug_dump_tensor_names:
        # Print a de-duplicated list to help user pick debug_output_tensors
        seen = set()
        names = []
        for t in _iter_all_tensors():
            try:
                n = str(t.name)
            except Exception:
                continue
            if not n or n in seen:
                continue
            seen.add(n)
            names.append(n)
        names.sort()
        logger.info("All network tensor names (dedup, %d):", len(names))
        for n in names:
            logger.info("  %s", n)
        print(colored(f"Dumped {len(names)} tensor names to log (debug_dump_tensor_names=True).", print_color))

    if debug_output_tensors:
        added = 0
        for n in debug_output_tensors:
            t = _find_tensor_by_name(n)
            if t is None:
                logger.warning("debug_output_tensors: tensor not found: %s", n)
                continue
            try:
                network.mark_output(t)
                added += 1
                logger.info("Marked debug output tensor: %s", t.name)
                print(colored(f"Marked debug output tensor: {t.name}", print_color))
            except Exception as exc:
                logger.warning("Failed to mark output %s: %s", n, exc)
        if added:
            logger.info("Network outputs after debug mark: %d", network.num_outputs)
            print(colored(f"Network outputs after debug mark: {network.num_outputs}", print_color))

    # Create builder config
    logger.info("\n[Step 3/5] Configuring builder...")
    print(colored("\n[Step 3/5] Configuring builder...", print_color))
    config = builder.create_builder_config()

    if use_cudagraph:
        config.set_flag(trt.BuilderFlag.CUDA_GRAPH)
        logger.info("Enabled CUDA_GRAPH mode")
        print(colored("Enabled CUDA_GRAPH mode", print_color))

    # Enable detailed profiling for engine inspection
    # This allows get_layer_information() to return layer types, precisions, tactics, etc.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    logger.info("Enabled DETAILED profiling verbosity for engine inspection")
    print(colored("Enabled DETAILED profiling verbosity for engine inspection", print_color))

    # Set workspace
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1024**2))

    # Set precision（STRONGLY_TYPED 网络禁止再开 FP16/BF16 等 Builder 混精标志）
    if strongly_typed_network:
        logger.info(
            "Skipping BuilderFlag FP16/BF16/FP8/INT4: STRONGLY_TYPED network "
            f"(precision hint from config: {precision!r} — actual types follow ONNX)"
        )
        print(
            colored(
                "Skipping global precision flags (STRONGLY_TYPED: types from ONNX)",
                print_color,
            )
        )
    else:
        # Precision constraint policy (useful when layer_precision_overrides is provided).
        pol = str(precision_constraints or "").strip().lower()
        if pol:
            if pol == "obey":
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                logger.info("Enabled OBEY_PRECISION_CONSTRAINTS")
                print(colored("Enabled OBEY_PRECISION_CONSTRAINTS", print_color))
            elif pol == "prefer":
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
                logger.info("Enabled PREFER_PRECISION_CONSTRAINTS")
                print(colored("Enabled PREFER_PRECISION_CONSTRAINTS", print_color))
            else:
                raise ValueError(
                    f"Unknown precision_constraints policy: {precision_constraints!r} "
                    "(expected 'obey' or 'prefer')"
                )

        tokens = normalize_precision_tokens(precision)
        if not tokens:
            raise ValueError(f"Empty precision: {precision!r}")
        if len(tokens) == 1:
            # Back-compat: allow legacy single-string values without treating commas.
            apply_builder_precision_flags(config, tokens)
        else:
            logger.info("Multi-token precision mode: %s", ",".join(tokens))
            print(colored(f"Multi-token precision mode: {','.join(tokens)}", print_color))
            apply_builder_precision_flags(config, tokens)

    # Set optimization profiles for dynamic shapes
    if min_shapes and opt_shapes and max_shapes:
        logger.info("\n[Step 4/5] Setting optimization profiles...")
        print(
            colored("\n[Step 4/5] Setting optimization profiles...", print_color))
        profile = builder.create_optimization_profile()

        for i in range(network.num_inputs):
            inp = network.get_input(i)
            input_name = inp.name

            if input_name in min_shapes:
                min_shape = min_shapes[input_name]
                opt_shape = opt_shapes[input_name]
                max_shape = max_shapes[input_name]

                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.info(f"  {input_name}:")
                logger.info(f"    min: {min_shape}")
                logger.info(f"    opt: {opt_shape}")
                logger.info(f"    max: {max_shape}")

        config.add_optimization_profile(profile)
    else:
        raise RuntimeError("Provide min/max and opt shapes for dynamic axes")

    # Build engine
    logger.info("\n[Step 5/5] Building TensorRT engine...")
    print(colored("\n[Step 5/5] Building TensorRT engine...", print_color))

    start_time = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - start_time

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    logger.info(
        f"Engine built in {build_time:.1f} seconds ({build_time / 60:.1f} minutes)")

    # Save engine
    logger.info(f"\nSaving engine to {engine_path}...")
    print(colored(f"\nSaving engine to {engine_path}...", print_color))
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = os.path.getsize(engine_path) / (1024**2)
    logger.info(f"Engine saved! Size: {engine_size_mb:.2f} MB")
    print(colored(f"Engine saved! Size: {engine_size_mb:.2f} MB", print_color))

    logger.info("\n" + "=" * 80)
    print(colored("\n" + "=" * 80, print_color))
    logger.info("ENGINE BUILD COMPLETE!")
    print(colored("ENGINE BUILD COMPLETE!", print_color))
    logger.info("=" * 80)
    print(colored("=" * 80, print_color))
    logger.info(f"Engine file: {engine_path}")
    print(colored(f"Engine file: {engine_path}", print_color))
    logger.info(f"Size: {engine_size_mb:.2f} MB")
    print(colored(f"Size: {engine_size_mb:.2f} MB", print_color))
    logger.info(f"Build time: {build_time:.1f}s")
    print(colored(f"Build time: {build_time:.1f}s", print_color))
    logger.info(f"Precision: {precision.upper()}")
    print(colored(f"Precision: {precision.upper()}", print_color))
    logger.info("=" * 80)
