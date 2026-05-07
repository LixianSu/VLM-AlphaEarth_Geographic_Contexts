# =====================================================================
# 终极版 SVI 语境语义提取与空间矢量化管线 (容错+监控+日志+Excel完整导出版)
# =====================================================================
import os
import glob
import json
import re
import pandas as pd  # 新增：用于生成 Excel
import geopandas as gpd
from shapely.geometry import Point
from qwen_vl_utils import process_vision_info
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

from main_entry.config import Config


def get_gpu_memory_usage():
    """实时获取当前进程占用的显存 (GB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 3
    return 0


def setup_controlled_vlm():
    """
    按需分配 GPU 资源并加载模型，避免挤占实验室公共算力。
    """
    gpu_ids = getattr(Config, 'gpu_id', [0])
    vram_limit_gb = getattr(Config, 'vram_limit_gb', 16)

    print(f"正在配置计算沙箱: 启用 GPU {gpu_ids}, 单卡显存上限: {vram_limit_gb}GB")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    max_mem_dict = {i: f"{vram_limit_gb}GiB" for i in range(len(gpu_ids))}

    print("正在按设定的资源配额加载 Qwen2.5-VL 7B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem_dict
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("模型受控加载完成！")

    return model, processor


def qwen25vl_to_shapefile(VLM_model, processor, gpu_device="cuda"):
    """
    读取指定文件夹下的街景，过滤残缺数据并生成日志，
    随后联合输出多尺度 Geographic Context，导出 Shapefile 与 完整 Excel。
    """
    folder_path = Config.CLEANED_SVI_DIR
    all_images = glob.glob(os.path.join(folder_path, "*.jpg"))
    location_groups = {}

    print("正在扫描并解析街景图像坐标...")
    # ==========================================
    # 步骤 1：防御性解析文件并提取经纬度坐标
    # ==========================================
    for img_path in all_images:
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('__')

        if len(parts) == 2:
            loc_id = parts[0]
            c = parts[1].split('_')

            if len(c) < 3:
                print(f"格式跳过: {filename} (参数不足)")
                continue

            try:
                lat = float(c[0])
                lon = float(c[1])
                heading = int(c[2])
            except ValueError:
                print(f"类型跳过: {filename} (包含非数字字符)")
                continue

            if loc_id not in location_groups:
                location_groups[loc_id] = {'coords': (lon, lat), 'images': []}

            location_groups[loc_id]['images'].append((heading, img_path))

    # ==========================================
    # 步骤 2：严格的数据过滤与缺失日志生成
    # ==========================================
    valid_location_groups = {}
    missing_logs = []

    shp_dir = os.path.join(folder_path, "shp")
    os.makedirs(shp_dir, exist_ok=True)
    log_file_path = os.path.join(shp_dir, "missing_SVI_points_log.txt")

    for loc_id, group_data in location_groups.items():
        if len(group_data['images']) != 4:
            lon, lat = group_data['coords']
            present_views = [t[0] for t in group_data['images']]
            missing_logs.append(f"Location ID: {loc_id} | Coords: ({lon}, {lat}) | 仅包含视角: {present_views}")
        else:
            valid_location_groups[loc_id] = group_data

    if missing_logs:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("=== SVI 数据视角缺失与剔除日志 ===\n")
            f.write(f"总计发现并剔除了 {len(missing_logs)} 个不完整采样点。\n\n")
            for log in missing_logs:
                f.write(log + "\n")
        print(f"发现 {len(missing_logs)} 个缺失视角的采样点。已剔除，详细日志保存至: {log_file_path}")
    else:
        print("\n所有采样点均具备完整的 4 个视角。")

    results_db = {}
    gdf_records = []     # 用于生成 Shapefile（长度截断）
    excel_records = []   # 用于生成 Excel（保留原始无限长文本）
    target_locations = list(valid_location_groups.keys())

    # ==========================================
    # 步骤 3：模型推理与空间记录构建
    # ==========================================
    pbar = tqdm(target_locations, desc="城市语境提取进度", unit="loc")

    for loc_id in pbar:
        group_data = valid_location_groups[loc_id]
        image_tuples = group_data['images']
        lon, lat = group_data['coords']

        mem_use = get_gpu_memory_usage()
        pbar.set_description(f"处理中: {loc_id} | VRAM: {mem_use:.2f}GB")

        image_tuples.sort(key=lambda x: x[0])
        sorted_img_paths = [t[1] for t in image_tuples]

        # 划分geographic Context的提示词
        """
        prompt_cop = As an urban geographer, comprehensively analyze the four continuous street view images provided above, which represent the same location from four different perspectives (front, right, back, left). Please output the multi-scale spatial ecological context of this location strictly in the following JSON format:
        {
            "micro_objects": "[Micro-level discrete entities, e.g., pushcarts, street lamp， piled goods, temporary stalls]",
            "meso_infrastructure": "[Meso-level infrastructure and topography, e.g., asphalt paved roads, stairs]",
            "macro_land_use": "[Macro-level land use and building function inference, e.g., ground-floor retail in podiums]",
            "spatial_relations": "[Interactive topological relationships among micro-entities, meso-infrastructure, and macro-functions]",
            "holistic_geographic_context": "[Based on the comprehensive observation of all perspectives, provide the ultimate qualitative description of the urban morphology and micro-ecology at this location]"
        }"""

        # 划分Land Use的提示词

        prompt_cop = """As an expert urban planner, comprehensively analyze the four continuous street view images provided above, which represent the same location from four different perspectives (front, right, back, left) in Hong Kong. 
        Your task is to conduct a multi-scale spatial reasoning process, and ultimately classify the dominant land use based strictly on the official Hong Kong Planning Department classification scheme.

        Please refer to the following hierarchy:
        1. Residential (Sub: Private Residential, Public Residential, Rural Settlement)
        2. Commercial (Sub: Commercial / Business and Office)
        3. Industrial (Sub: Industrial Land, Industrial Estates / Science Parks, Warehouse and Open Storage)
        4. Institutional / Open Space (Sub: Government/Institutional/Community Facilities, Open Space and Recreation)
        5. Transportation (Sub: Roads and Transport Facilities, Railways, Port Facilities)
        6. Other Urban or Built-up Land (Sub: Cemeteries, Utilities, Vacant Land)
        7. Natural / Rural (Sub: Agriculture, Woodland / Shrubland, Water Bodies)

        Strictly output the classification in the following JSON format without any markdown formatting:
        {
            "micro_objects": "[Micro-level discrete entities, e.g., pushcarts, street lamps, piled goods, temporary stalls]",
            "meso_infrastructure": "[Meso-level infrastructure and topography, e.g., asphalt paved roads, stairs]",
            "macro_land_use": "[Macro-level land use and building function inference, e.g., ground-floor retail in podiums]",
            "spatial_relations": "[Interactive topological relationships among micro-entities, meso-infrastructure, and macro-functions]",
            "holistic_context": "[A comprehensive paragraph summarizing the unique urban morphology and micro-ecology here, vital for downstream embedding]",
            "official_main_class": "[Must be exactly one of the 7 main categories listed above]",
            "official_sub_class": "[Must be the exact matching sub-category from the list above]"
        }"""

        content_list = [{"type": "image", "image": f"file://{img_path}"} for img_path in sorted_img_paths]
        content_list.append({"type": "text", "text": prompt_cop})
        messages = [{"role": "user", "content": content_list}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(gpu_device)

        with torch.no_grad():
            generated_ids = VLM_model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        # ==========================================
        # 步骤 4：JSON 防御性解析
        # ==========================================
        try:
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
            else:
                raise ValueError("未找到有效的 JSON 结构")

        except Exception as e:
            parsed_json = {
                "micro_objects": "Parse Failed",
                "meso_infrastructure": "Parse Failed",
                "macro_land_use": "Parse Failed",
                "spatial_relations": "Parse Failed",
                "holistic_context": "Parse Failed",
                "official_main_class": "Parse Failed",
                "official_sub_class": "Parse Failed"
            }

        # 对于Geographic contexts版本的提示词的检验
        '''except Exception as e:
            parsed_json = {
                "micro_objects": "Parse Failed", "meso_infrastructure": "Parse Failed",
                "macro_land_use": "Parse Failed", "spatial_relations": "Parse Failed",
                "holistic_geographic_context": "Parse Failed"
            }'''



        results_db[loc_id] = parsed_json
        """
        # 提取完整的字符串内容， 这个是划分geographic context的提示词配套使用
        full_mic_obj = str(parsed_json.get("micro_objects", ""))
        full_mes_inf = str(parsed_json.get("meso_infrastructure", ""))
        full_mac_lu = str(parsed_json.get("macro_land_use", ""))
        full_spa_rel = str(parsed_json.get("spatial_relations", ""))
        full_hol_ctx = str(parsed_json.get("holistic_geographic_context", ""))

        # 1. 存入 GIS 记录表 (Shapefile 必须截断至 250 字符以防崩溃)
        gdf_records.append({
            "loc_id": loc_id,
            "geometry": Point(lon, lat),
            "mic_obj": full_mic_obj[:250],
            "mes_inf": full_mes_inf[:250],
            "mac_lu": full_mac_lu[:250],
            "spa_rel": full_spa_rel[:250],
            "hol_ctx": full_hol_ctx[:250]
        })

        # 2. 存入 Excel 记录表 (保留坐标与原始、无限长的完整文本)
        excel_records.append({
            "Location_ID": loc_id,
            "Longitude": lon,
            "Latitude": lat,
            "Micro_Objects": full_mic_obj,
            "Meso_Infrastructure": full_mes_inf,
            "Macro_Land_Use": full_mac_lu,
            "Spatial_Relations": full_spa_rel,
            "Holistic_Context": full_hol_ctx
        })
        """

        # ==========================================
        # 步骤 4：JSON 防御性解析与数据挂载 (严格对齐Land Use版本)
        # ==========================================
        # 提取基础空间生态语义
        full_mic_obj = str(parsed_json.get("micro_objects", ""))
        full_mes_inf = str(parsed_json.get("meso_infrastructure", ""))
        full_mac_lu = str(parsed_json.get("macro_land_use", ""))
        full_spa_rel = str(parsed_json.get("spatial_relations", ""))

        # 提取极其重要的“语义黄金”（键名已改为 holistic_context）
        full_hol_ctx = str(parsed_json.get("holistic_context", ""))

        # 提取新增的 HK PlanD 官方分类
        full_pri_lu = str(parsed_json.get("official_main_class", ""))
        full_sec_lu = str(parsed_json.get("official_sub_class", ""))

        # 1. 存入 GIS 记录表 (Shapefile 要求：字段名极短且不超过10字符，文本必须截断至250字符防爆)
        gdf_records.append({
            "loc_id": loc_id,
            "geometry": Point(lon, lat),
            "mic_obj": full_mic_obj[:250],
            "mes_inf": full_mes_inf[:250],
            "mac_lu": full_mac_lu[:250],
            "spa_rel": full_spa_rel[:250],
            "hol_ctx": full_hol_ctx[:250],
            "pri_lu": full_pri_lu[:250],  # 新增装载：一级土地分类
            "sec_lu": full_sec_lu[:250]  # 新增装载：二级土地分类
        })

        # 2. 存入 Excel 记录表 (无限制容器：保留坐标与原始、无限长的完整文本)
        excel_records.append({
            "Location_ID": loc_id,
            "Longitude": lon,
            "Latitude": lat,
            "Micro_Objects": full_mic_obj,
            "Meso_Infrastructure": full_mes_inf,
            "Macro_Land_Use": full_mac_lu,
            "Spatial_Relations": full_spa_rel,
            "Holistic_Context": full_hol_ctx,
            "Official_Main_Class": full_pri_lu,  # 新增装载：一级土地分类
            "Official_Sub_Class": full_sec_lu  # 新增装载：二级土地分类
        })

    # ==========================================
    # 步骤 5：生成并保存 Shapefile 与 Excel
    # ==========================================
    # 保存 Shapefile
    if gdf_records:
        gdf = gpd.GeoDataFrame(gdf_records, crs="EPSG:4326")
        out_shp_path = os.path.join(shp_dir, "SVI_Context_Results.shp")
        gdf.to_file(out_shp_path, driver='ESRI Shapefile', encoding='utf-8')
        print(f"\n 成功！空间矢量文件已保存至: {out_shp_path}")
    else:
        print("\n 警告：未能生成任何有效矢量记录。")

    # 保存 Excel
    if excel_records:
        df = pd.DataFrame(excel_records)
        out_excel_path = os.path.join(shp_dir, "SVI_Context_Results.xlsx")
        df.to_excel(out_excel_path, index=False)
        print(f" 成功！完整属性记录（Excel）已保存至: {out_excel_path}")
    else:
        print(" 警告：未能生成任何有效 Excel 记录。")

    return results_db