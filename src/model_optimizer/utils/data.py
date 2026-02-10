import numpy as np
def load_saved_data(input_data_file: str):
    """
    从npz文件加载字典列表
    """
    loaded = np.load(input_data_file, allow_pickle=True)

    # 提取元数据
    meta = loaded['metadata'].item()
    num_items = meta['num_items']

    # 重建字典列表
    result = []
    for i in range(num_items):
        item_dict = {}
        # 查找属于当前item的所有键
        for key in loaded.files:
            if key.startswith(f"item_{i}_"):
                original_key = key.replace(f"item_{i}_", "", 1)
                item_dict[original_key] = loaded[key]

        if item_dict:  # 只添加非空字典
            result.append(item_dict)

    loaded.close()
    return result