# =====================================================================
# 终极版 SVI 语境语义提取与空间矢量化管线 (导出 Shapefile) Zero Shot
# =====================================================================
import os
import glob
import json
import re
import geopandas as gpd
from main_entry.config import Config
from shapely.geometry import Point
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# ---------------------------------------------------------
# 【全局初始化】
# ---------------------------------------------------------
print("正在初始化 Qwen2.5-VL 7B 与 Processor (分配至双 4090D)...")

def qwen25vl_to_shapefile():
    """
    读取指定文件夹下的街景，联合输出多尺度 Geographic Context，
    并自动将结果导出为包含空间坐标与丰富语义属性的 Shapefile。
    """
    VLM_model = Config.VLM_model
    processor = Config.processor
    folder_path = Config.CLEANED_SVI_DIR

    all_images = glob.glob(os.path.join(folder_path, "*.jpg"))
    location_groups = {}

    # ==========================================
    # 步骤 1：解析文件并提取经纬度坐标
    # 示例: ID1__22.28378_113.94246_90.jpg
    # ==========================================
    for img_path in all_images:
        filename = os.path.basename(img_path)
        name_without_ext = filename.split('.')[0]
        parts = name_without_ext.split('__')

        if len(parts) == 2:
            loc_id = parts[0]
            coords_and_heading = parts[1].split('_')

            try:
                lat = float(coords_and_heading[0])
                lon = float(coords_and_heading[1])
                heading = int(coords_and_heading[2])
            except ValueError:
                continue

            if loc_id not in location_groups:
                # 将坐标也记录下来，用于构建 Geometry
                location_groups[loc_id] = {'coords': (lon, lat), 'images': []}

            location_groups[loc_id]['images'].append((heading, img_path))

    results_db = {}
    gdf_records = []  # 用于构建 GeoDataFrame 的记录列表

    # ==========================================
    # 步骤 2：模型推理与空间记录构建
    # ==========================================
    for loc_id, group_data in location_groups.items():
        image_tuples = group_data['images']
        lon, lat = group_data['coords']

        if len(image_tuples) != 4:
            print(f"警告：地点 {loc_id} 的图像数量不是4张，跳过。")
            continue

        print(f"正在分析地点: {loc_id} 的微观生态语境...")

        image_tuples.sort(key=lambda x: x[0])
        sorted_img_paths = [t[1] for t in image_tuples]

        prompt_cop = """请作为城市地理学家，综合分析以上四张代表同一地点（前方、右侧、后方、左侧）的连续街景图像，严格按以下JSON格式输出该地点的多尺度空间生态语境：
        {
            "micro_objects": "[微观离散实体，如：手推车、货物堆放、临时摊位]",
            "meso_infrastructure": "[中观基础设施与地形，如：沥青硬化路面、阶梯]",
            "macro_land_use": "[宏观土地利用与建筑功能推断，如：底层裙楼商业]",
            "spatial_relations": "[微观实体、中观基建与宏观功能之间的互动拓扑关系]",
            "holistic_geographic_context": "[综合以上视角的观察，输出对该地点城市形态与微观生态的终极定性描述]"
        }"""

        content_list = [{"type": "image", "image": f"file://{img_path}"} for img_path in sorted_img_paths]
        content_list.append({"type": "text", "text": prompt_cop})
        messages = [{"role": "user", "content": content_list}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda")

        generated_ids = VLM_model.generate(**inputs, max_new_tokens=1024, temperature=0.0)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        # ==========================================
        # 步骤 3：JSON 防御性解析 (清洗大模型的格式杂质)
        # ==========================================
        try:
            # 匹配大括号中的内容，防止模型输出 Markdown 标记如 ```json ... ```
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
            else:
                raise ValueError("未找到有效的 JSON 结构")
        except Exception as e:
            print(f"JSON 解析失败 (地点 {loc_id}): {e}")
            parsed_json = {
                "micro_objects": "解析失败", "meso_infrastructure": "解析失败",
                "macro_land_use": "解析失败", "spatial_relations": "解析失败",
                "holistic_geographic_context": "解析失败"
            }

        results_db[loc_id] = parsed_json

        # 构建 GIS 记录表 (使用 10 字符以内的缩写字段名)
        gdf_records.append({
            "loc_id": loc_id,
            "geometry": Point(lon, lat),
            "mic_obj": parsed_json.get("micro_objects", ""),
            "mes_inf": parsed_json.get("meso_infrastructure", ""),
            "mac_lu": parsed_json.get("macro_land_use", ""),
            "spa_rel": parsed_json.get("spatial_relations", ""),
            "hol_ctx": parsed_json.get("holistic_geographic_context", "")
        })

    # ==========================================
    # 步骤 4：生成并保存 Shapefile
    # ==========================================
    if gdf_records:
        # 创建输出文件夹
        shp_dir = os.path.join(folder_path, "shp")
        os.makedirs(shp_dir, exist_ok=True)

        # 将记录转化为 GeoDataFrame，并设定坐标系为 WGS84 (EPSG:4326)
        gdf = gpd.GeoDataFrame(gdf_records, crs="EPSG:4326")

        # 导出 Shapefile (必须指定 utf-8 编码，否则中文字符在 ArcGIS 中会乱码)
        out_shp_path = os.path.join(shp_dir, "SVI_Context_Results.shp")
        gdf.to_file(out_shp_path, driver='ESRI Shapefile', encoding='utf-8')
        print(f"\n✅ 成功！空间矢量文件已保存至: {out_shp_path}")
    else:
        print("\n❌ 警告：未能生成任何有效记录，Shapefile 创建失败。")

    return results_db

# -----------------
# 运行调用示例
# -----------------
# final_results = qwen25vl_to_shapefile(model, processor)