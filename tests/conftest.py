"""pytest 根级 conftest，提供全局 fixtures。"""

import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
