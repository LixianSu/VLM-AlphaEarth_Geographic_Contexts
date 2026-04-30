import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from config import Config


def clean_and_rename_street_views():
    """
    清洗街景数据：仅保留四个角度皆存在的地理位置，并统一重命名。
    输出格式: ID[自增编号]__[纬度]_[经度]_[朝向].[扩展名]
    """
    print(">>> 开始扫描原始街景数据...")

    # 结构: coordinate_dict["22.47666476_114.1746823"] = { "0": Path(...), "90": Path(...), ... }
    coordinate_dict = defaultdict(dict)

    # 1. 遍历四个角度的文件夹，提取文件信息
    for angle_folder in Config.SVI_ANGLE_FOLDERS:
        # 从文件夹名 "HK_0" 提取朝向 "0"
        angle = angle_folder.split("_")[1]
        folder_path = Config.BASE_DATA_DIR / angle_folder

        if not folder_path.exists():
            print(f"警告: 找不到文件夹 {folder_path}")
            continue

        for img_path in folder_path.glob("*.*"):  # 匹配所有图片文件
            # 文件名示例: 500_31750__22.47666476_114.1746823.jpg
            filename = img_path.stem
            ext = img_path.suffix

            # 按照 "__" 分割，取后半部分的经纬度
            parts = filename.split("__")
            if len(parts) != 2:
                continue

            lat_lon_str = parts[1]  # 例如 "22.47666476_114.1746823"

            try:
                # 进一步分割纬度和经度
                lat_str, lon_str = lat_lon_str.split("_")

                # 【核心修改】：引入空间容差
                # 保留5位小数，约等于现实中的 1.1 米误差范围
                # 保留4位小数，约等于现实中的 11 米误差范围
                ROUNDING_DECIMALS = 5

                lat = round(float(lat_str), ROUNDING_DECIMALS)
                lon = round(float(lon_str), ROUNDING_DECIMALS)

                # 重新组合成带有容差的统一坐标 Key
                tolerance_key = f"{lat}_{lon}"

                # 将该图片归入这个带容差的坐标聚类中，并保留原始文件名供后续复制
                coordinate_dict[tolerance_key][angle] = img_path

            except ValueError:
                # 捕获可能的格式错误（如非数字字符）
                continue

            lat_lon = parts[1]  # 例如 "22.47666476_114.1746823"
            coordinate_dict[lat_lon][angle] = img_path

    # 2. 筛选并复制符合条件的文件
    valid_locations = 0
    new_id_counter = 1

    print(f">>> 扫描完成，共发现 {len(coordinate_dict)} 个独立的坐标点。")
    print(">>> 开始执行交叉验证与清洗转移...")

    for lat_lon, angle_paths in tqdm(coordinate_dict.items(), desc="处理进度"):
        # 严格检查：是否四个角度的文件都存在？
        if len(angle_paths) == 4:
            valid_locations += 1
            new_location_id = f"ID{new_id_counter}"

            # 遍历四个角度的原始文件路径，进行复制和重命名
            for angle, original_path in angle_paths.items():
                ext = original_path.suffix
                # 构造新文件名: ID1__22.47666476_114.1746823_0.jpg
                new_filename = f"{new_location_id}__{lat_lon}_{angle}{ext}"
                target_path = Config.CLEANED_SVI_DIR / new_filename

                # 执行复制 (使用 copy2 保留原始文件的元数据)
                shutil.copy2(original_path, target_path)

            new_id_counter += 1

    print(f"\n>>> 清洗任务完成！")
    print(f">>> 共有 {valid_locations} 个位置满足全角度条件。")
    print(f">>> 最终保留了 {valid_locations * 4} 张有效街景图片。")
    print(f">>> 文件已保存至: {Config.CLEANED_SVI_DIR}")


if __name__ == "__main__":
    # 允许直接运行此脚本进行测试
    Config.setup_directories()
    clean_and_rename_street_views()