import os
from pathlib import Path


class Config:
    """
    SVI-AE 跨模态地理空间对齐与发现框架 - 全局配置中心
    """
    # ==========================================
    # 1. 基础路径配置 (Directory Paths)
    # ==========================================
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

    # 输入文件路径 (确保在此处放入你的原始数据)
    SVI_SHP_PATH = DATA_DIR / "raw" / "Singapore_SVI_points.shp"
    ROAD_NETWORK_PATH = DATA_DIR / "raw" / "Singapore_Roads.shp"
    AE_RASTER_PATH = DATA_DIR / "raw" / "AlphaEarth_SG_10m.tif"

    # 模型输出与检查点路径
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    MAP_OUTPUT_DIR = OUTPUT_DIR / "maps"

    # ==========================================
    # 2. VLM 与文本特征参数 (VLM & Text Encoder Params)
    # ==========================================
    # 注意：在实际运行前，请在终端设置环境变量，切勿将真实 API Key 明文写在代码里
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-fallback-key")
    VLM_MODEL = "gpt-4o"
    TEXT_ENCODER_MODEL = "text-embedding-3-large"
    TEXT_FEATURE_DIM = 384  # 根据 text-embedding 模型的输出维度设定

    # ==========================================
    # 3. GIS 空间分析与池化参数 (Spatial & Pooling Params)
    # ==========================================
    # 对应 AETHER 论文中的多尺度视图设计
    RADIUS_BASE = 50  # 基础视图半径 (米) - 捕捉局部精细形态
    RADIUS_AUG = 150  # 增强视图半径 (米) - 捕捉宏观街区上下文

    # 解决空间错位的核心参数
    MAX_NETWORK_DIST = 200  # 街景语义最大有效路网辐射距离 (米)
    DECAY_RATE = 0.02  # 指数衰减系数 (决定了距离街道越远，语义权重下降的速度)

    # ==========================================
    # 4. 深度学习与 Residual-SE 模型参数 (DL & Network Params)
    # ==========================================
    # 输入与潜空间维度 (结合 Zhang et al. 2026 的设计)
    AE_INPUT_DIM = 64  # AlphaEarth 原始输入维度
    LATENT_DIM = 128  # 最终跨模态对齐的潜空间维度 (斩断后的输出层)

    # 训练超参数
    BATCH_SIZE = 512
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 正则化，防止过拟合

    # 跨模态对比学习 (InfoNCE) 温度与权重参数
    TAU_AC = 0.07  # 模态内 (AE-AE) 多尺度一致性温度参数
    TAU_POI = 0.07  # 跨模态 (AE-SVI) 对齐温度参数
    LAMBDA_WEIGHT = 0.2  # 联合损失函数的平衡权重 (L_AA 与 L_AS 之间的平衡)

    # ==========================================
    # 5. 无监督发现阶段参数 (Discovery Pipeline Params)
    # ==========================================
    # HDBSCAN 聚类参数
    MIN_CLUSTER_SIZE = 50  # 认定为一个有效“城市语境”的最少网格数
    MIN_SAMPLES = 15  # 控制对噪声(Noise)的容忍度，值越大，被归为噪声的网格越多

    # 语义回溯提取样本量
    MEDOIDS_PER_CLUSTER = 50  # 每次让 GPT-4o 归纳概念时，提取的最具代表性的街景数量

    @classmethod
    def setup_directories(cls):
        """运行前自动创建缺失的文件夹"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (cls.DATA_DIR / "raw").mkdir(exist_ok=True)
        (cls.DATA_DIR / "processed").mkdir(exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 可以在 main.py 中这样使用：
# from config import Config
# Config.setup_directories()