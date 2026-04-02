import re

import numpy as np

# item_<非负整数>_<原始键名>；原始键名可含下划线。
# 不能用 startswith(f"item_{i}_")：例如 item_10_* 会误匹配 item_1_*，item_10_* 也会误匹配 item_0_*。
_ITEM_KEY_RE = re.compile(r"^item_(\d+)_(.+)$")


def load_saved_data(input_data_file: str):
    """
    从npz文件加载字典列表
    """
    loaded = np.load(input_data_file, allow_pickle=True)

    # 提取元数据
    meta = loaded['metadata'].item()
    num_items = meta['num_items']

    by_index: dict[int, dict[str, object]] = {}
    for key in loaded.files:
        if key == "metadata":
            continue
        m = _ITEM_KEY_RE.match(key)
        if not m:
            continue
        idx = int(m.group(1))
        original_key = m.group(2)
        by_index.setdefault(idx, {})[original_key] = loaded[key]

    result = []
    for i in range(num_items):
        item_dict = by_index.get(i, {})
        if item_dict:
            result.append(item_dict)

    loaded.close()
    return result