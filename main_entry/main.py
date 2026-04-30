import main_entry.config
from vlm_processor import process_svi_to_semantics
from spatial_ops import extract_and_aggregate_features
from dataset import SVI_AE_Dataset, create_dataloader
from networks import MultimodalProjector
from trainer import ModelTrainer


def main():
    """
    SVI-AE 跨模态地理空间对齐框架 - 主控入口
    """
    print(">>> 正在启动 SVI-AE 跨模态对齐实验流...")

    # ==========================================
    # 步骤 0: 加载全局配置
    # ==========================================
    print(">>> 步骤 0: 初始化全局配置...")
    cfg = config.load_config()

    # ==========================================
    # 步骤 1: 视觉语言特征提取 (阶段一)
    # ==========================================
    print(">>> 步骤 1: 提取街景影像的 VLM 文本描述与高维向量...")
    # 这里会调用大模型生成文本，并转换为语义向量
    svi_semantics = process_svi_to_semantics(
        svi_path=cfg.SVI_DATA_PATH,
        model_name=cfg.VLM_MODEL_NAME
    )

    # ==========================================
    # 步骤 2 & 3: GIS 空间分析与距离衰减聚合
    # ==========================================
    print(">>> 步骤 2 & 3: 提取 AlphaEarth 特征并执行距离衰减加权池化...")
    # 这一步是耗时大户，负责处理所有的栅格与矢量路网拓扑计算
    aligned_data = extract_and_aggregate_features(
        ae_raster_path=cfg.AE_RASTER_PATH,
        svi_semantics=svi_semantics,
        base_radius=cfg.RADIUS_BASE,
        aug_radius=cfg.RADIUS_AUG,
        max_dist=cfg.MAX_NETWORK_DIST
    )

    # ==========================================
    # 步骤 4: 构建 PyTorch 数据集
    # ==========================================
    print(">>> 步骤 4: 准备深度学习训练数据集...")
    train_dataset = SVI_AE_Dataset(aligned_data)
    train_loader = create_dataloader(train_dataset, batch_size=cfg.BATCH_SIZE)

    # ==========================================
    # 步骤 5: 初始化多模态对比学习网络 (阶段四)
    # ==========================================
    print(">>> 步骤 5: 初始化跨模态对齐网络 (MLP Projectors)...")
    # 确保 AE 的物理特征维度与 SVI 的文本特征维度正确映射到潜空间
    model = MultimodalProjector(
        ae_dim=cfg.AE_FEATURE_DIM,
        text_dim=cfg.TEXT_FEATURE_DIM,
        latent_dim=cfg.LATENT_DIM
    )

    # ==========================================
    # 步骤 6: 启动训练引擎
    # ==========================================
    print(">>> 步骤 6: 启动模型训练...")
    trainer = ModelTrainer(
        model=model,
        dataloader=train_loader,
        learning_rate=cfg.LEARNING_RATE,
        epochs=cfg.EPOCHS,
        tau_ac=cfg.TAU_AC,
        tau_poi=cfg.TAU_POI
    )

    # 执行训练并保存最优模型权重
    trainer.train()
    print(">>> 实验完成！模型权重已保存至输出目录。")


if __name__ == '__main__':
    main()