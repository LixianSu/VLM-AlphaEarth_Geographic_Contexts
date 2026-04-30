# 从当前目录下的各个子文件，导入核心类和函数
from .networks import MultimodalProjector
from .losses import info_nce_loss
from .trainer import ModelTrainer

# __all__ 是一个非常专业的 Python 规范
# 它明确规定了：当别人使用 `from core_models import *` 时，到底哪些东西会被导出去。
# 这能有效防止内部的临时变量或辅助库（比如 import torch）污染外部的命名空间。
__all__ = [
    "MultimodalProjector",
    "info_nce_loss",
    "ModelTrainer"
]