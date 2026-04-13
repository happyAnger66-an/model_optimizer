"""可选配置文件：``--webui-config`` 将 YAML/JSON 中的键展开为 tyro 参数（用户命令行在后，可覆盖文件）。"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

import tyro

logger = logging.getLogger(__name__)

T = TypeVar("T")

_WEBUI_CONFIG_FLAGS = ("--webui-config", "--webui_config")


def _strip_webui_config_from_argv(argv: list[str]) -> tuple[list[str], Path | None]:
    """从 argv 中移除 ``--webui-config[=]PATH``，返回 (新 argv, 路径或 None)。"""
    out: list[str] = []
    i = 0
    path: Path | None = None
    while i < len(argv):
        a = argv[i]
        hit = False
        for key in _WEBUI_CONFIG_FLAGS:
            if a == key:
                if i + 1 >= len(argv):
                    raise ValueError(f"{key} 需要紧跟配置文件路径")
                path = Path(argv[i + 1]).expanduser()
                i += 2
                hit = True
                break
            eq = f"{key}="
            if a.startswith(eq):
                path = Path(a[len(eq) :]).expanduser()
                i += 1
                hit = True
                break
        if hit:
            continue
        out.append(a)
        i += 1
    return out, path


def _load_mapping(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    text = path.read_text(encoding="utf-8")
    suf = path.suffix.lower()
    if suf in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("读取 YAML 需要安装 PyYAML：pip install pyyaml") from exc
        data = yaml.safe_load(text)
    elif suf == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"不支持的配置文件后缀 {path.suffix!r}，请使用 .yaml / .yml / .json")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"配置文件根节点必须是 mapping/dict，实际为 {type(data).__name__}")
    return data


def _is_optional(typ: Any) -> bool:
    args = get_args(typ)
    return type(None) in args


def _field_cli_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _bool_argv_tokens(name: str, val: bool) -> list[str]:
    flag = _field_cli_flag(name)
    return [flag] if val else [f"--no-{name.replace('_', '-')}"]


def _value_to_argv_tokens(field_name: str, val: Any, field_type: Any) -> list[str]:
    """将单个字段值转为 tyro 可解析的 argv token 序列。"""
    flag = _field_cli_flag(field_name)
    if val is None:
        if _is_optional(field_type):
            return [flag, "None"]
        raise TypeError(f"{field_name}: 不能为 null（该字段非 Optional）")

    origin = get_origin(field_type)
    if field_type is bool:
        if not isinstance(val, bool):
            raise TypeError(f"{field_name}: 期望 bool，实际为 {type(val).__name__}")
        return _bool_argv_tokens(field_name, val)

    if isinstance(val, bool):
        return _bool_argv_tokens(field_name, val)

    if origin in (tuple, list):
        if not isinstance(val, (list, tuple)):
            raise TypeError(f"{field_name}: 期望 list/tuple，实际为 {type(val).__name__}")
        seq = list(val)
        if len(seq) == 0:
            return []
        return [flag, *[str(x) for x in seq]]

    if isinstance(val, Path):
        return [flag, str(val)]
    if isinstance(val, (str, int, float)):
        return [flag, str(val)]
    return [flag, str(val)]


def config_dict_to_argv(cfg_cls: type[T], data: dict[str, Any]) -> list[str]:
    """将扁平 dict（键为 ``Args`` 的字段名 snake_case）转为 tyro argv；未知键告警并忽略。"""
    fields = {f.name: f for f in dataclasses.fields(cfg_cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        logger.warning("配置文件中有未识别的字段，已忽略: %s", ", ".join(unknown))
    out: list[str] = []
    for f in dataclasses.fields(cfg_cls):
        if f.name not in data:
            continue
        out.extend(_value_to_argv_tokens(f.name, data[f.name], f.type))
    return out


def parse_args_with_optional_config_file(cfg_cls: type[T]) -> T:
    """等价于 ``tyro.cli(cfg_cls)``，但若命令行含 ``--webui-config`` 则先加载 YAML/JSON 并前置为默认参数。"""
    import sys

    argv = list(sys.argv[1:])
    argv, cfg_path = _strip_webui_config_from_argv(argv)
    merged = argv
    if cfg_path is not None:
        raw = _load_mapping(cfg_path)
        merged = config_dict_to_argv(cfg_cls, raw) + argv
    return tyro.cli(cfg_cls, args=merged)
