from config import Config
from data_processing.street_view_cleaning import clean_and_rename_street_views
from VLLM.qwen25vl import qwen25vl_to_shapefile
def main():
    """
    SVI-AE 跨模态地理空间对齐框架 - 主控入口
    当前阶段：第一阶段 - 街景数据基础清洗
    """
    print("==================================================")
    print("   SVI-AE Project - 街景数据预处理流水线启动   ")
    print("==================================================")

    # 步骤 0: 初始化输出目录
    Config.setup_directories()

    # 步骤 1: 街景数据清洗与重命名
    clean_and_rename_street_views()

    # 步骤 2： 读取街景数据进行分析
    qwen25vl_to_shapefile()

    print("\n>>> 当前阶段任务已全部执行完毕。等待进入 VLM 语义提取阶段。")

if __name__ == '__main__':
    main()