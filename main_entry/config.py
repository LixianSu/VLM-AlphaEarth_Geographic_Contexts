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
    BASE_DATA_DIR = Path(r"D:\Su_Lixian\Data\Google Street View")

    # 原始四个角度街景的文件夹名称
    SVI_ANGLE_FOLDERS = ["HK_0", "HK_90", "HK_180", "HK_270"]

    # CLEANED_SVI_DIR，并匹配你本地的 "Cleaned" 大小写
    CLEANED_SVI_DIR = BASE_DATA_DIR / "Cleaned"

    # ==========================================
    # 2. 模型静态配置 (The Blueprint)
    # ==========================================
    # 不在这里 import transformers 和加载模型
    # 只保留模型名称，主程序去负责实际的物理加载。
    gpu_id = [0] # 使用第一张gpu

    vram_limit_gb = 16 # 限制使用16GB内存

    VLM_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    # 也可以把后续空间池化相关的超参数静态地放在这里
    BUFFER_RADIUS_M = 50  # 欧氏距离缓冲区半径（米）

    @classmethod
    def setup_directories(cls):
        """运行前自动创建缺失的文件夹"""
        cls.CLEANED_SVI_DIR.mkdir(parents=True, exist_ok=True)