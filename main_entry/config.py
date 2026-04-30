import os
from pathlib import Path


class Config:
    """
    SVI-AE 跨模态地理空间对齐与发现框架 - 全局配置中心
    """
    # ==========================================
    # 1. 基础路径配置 (Directory Paths)
    # ==========================================
    # 针对你本地环境的绝对路径
    BASE_DATA_DIR = Path(r"E:\Data\Hong Kong\Street view images\Street View 100m\All")

    # 原始四个角度街景的文件夹名称
    SVI_ANGLE_FOLDERS = ["HK_0", "HK_90", "HK_180", "HK_270"]

    # 清洗后的输出路径
    CLEANED_SVI_DIR = BASE_DATA_DIR / "cleaned"

    # ==========================================
    # 2. 模型与其他配置 (为后续阶段保留，当前暂时挂起)
    # ==========================================
    # 这些是我们之前讨论过的参数，保留在这里以维持配置文件的完整性
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-fallback-key")
    VLM_MODEL = "gpt-4o"
    RADIUS_BASE = 50
    RADIUS_AUG = 150
    MAX_NETWORK_DIST = 200

    @classmethod
    def setup_directories(cls):
        """运行前自动创建缺失的文件夹"""
        cls.CLEANED_SVI_DIR.mkdir(parents=True, exist_ok=True)