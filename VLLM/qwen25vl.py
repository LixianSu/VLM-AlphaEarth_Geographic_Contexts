# =====================================================================
# 终极版 SVI 语境语义提取与空间矢量化管线 (容错+监控+日志记录版)
# =====================================================================
import os
import glob
import json
import re
import geopandas as gpd
from shapely.geometry import Point
from qwen_vl_utils import process_vision_info
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# 确保你的 config.py 已经按照我们之前的讨论完成了轻量化重构
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
    # 假设你的 Config 中定义了 gpu_id=[0] 和 vram_limit_gb=16
    gpu_ids = getattr(Config, 'gpu_id', [0])
    vram_limit_gb = getattr(Config, 'vram_limit_gb', 16)

    print(f"🔧 正在配置计算沙箱: 启用 GPU {gpu_ids}, 单卡显存上限: {vram_limit_gb}GB")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    max_mem_dict = {i: f"{vram_limit_gb}GiB" for i in range(len(gpu_ids))}

    print("⏳ 正在按设定的资源配额加载 Qwen2.5-VL 7B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem_dict
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("✅ 模型受控加载完成！")

    return model, processor


def qwen25vl_to_shapefile(VLM_model, processor, gpu_device="cuda"):
    """
    读取指定文件夹下的街景，过滤残缺数据并生成日志，
    随后联合输出多尺度 Geographic Context，导出 Shapefile。
    """
    folder_path = Config.CLEANED_SVI_DIR
    all_images = glob.glob(os.path.join(folder_path, "*.jpg"))
    location_groups = {}

    print("📂 正在扫描并解析街景图像坐标...")
    # ==========================================
    # 步骤 1：防御性解析文件并提取经纬度坐标
    # ==========================================
    for img_path in all_images:
        filename = os.path.basename(img_path)
        name_without_ext = filename.split('.')[0]
        parts = name_without_ext.split('__')

        if len(parts) == 2:
            loc_id = parts[0]
            c = parts[1].split('_')

            # 🛑 核心防御机制：校验文件名分段数量
            if len(c) < 3:
                print(f"⚠️ 格式跳过: {filename} (参数不足)")
                continue

            try:
                lat = float(c[0])
                lon = float(c[1])
                heading = int(c[2])
            except ValueError:
                print(f"⚠️ 类型跳过: {filename} (包含非数字字符)")
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

    # 写入日志
    if missing_logs:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("=== SVI 数据视角缺失与剔除日志 ===\n")
            f.write(f"总计发现并剔除了 {len(missing_logs)} 个不完整采样点。\n\n")
            for log in missing_logs:
                f.write(log + "\n")
        print(f"⚠️ 发现 {len(missing_logs)} 个缺失视角的采样点。已剔除，日志保存至: {log_file_path}")
    else:
        print("✅ 完美！所有采样点均具备完整的 4 个视角。")

    results_db = {}
    gdf_records = []
    target_locations = list(valid_location_groups.keys())

    # ==========================================
    # 步骤 3：模型推理与空间记录构建 (加入 tqdm 监控)
    # ==========================================
    pbar = tqdm(target_locations, desc="🏮 城市语境提取进度", unit="loc")

    for loc_id in pbar:
        group_data = valid_location_groups[loc_id]
        image_tuples = group_data['images']
        lon, lat = group_data['coords']

        # 动态更新进度条显示显存占用
        mem_use = get_gpu_memory_usage()
        pbar.set_description(f"🏮 处理中: {loc_id} | VRAM: {mem_use:.2f}GB")

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
        ).to(gpu_device)

        # 执行推理：使用 no_grad 阻断梯度图构建，极大节省显存
        with torch.no_grad():
            generated_ids = VLM_model.generate(**inputs, max_new_tokens=1024, temperature=0.0)

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
                "micro_objects": "解析失败", "meso_infrastructure": "解析失败",
                "macro_land_use": "解析失败", "spatial_relations": "解析失败",
                "holistic_geographic_context": "解析失败"
            }

        results_db[loc_id] = parsed_json

        # 截断字符串以防超出 ESRI Shapefile 254字符 的属性限制
        gdf_records.append({
            "loc_id": loc_id,
            "geometry": Point(lon, lat),
            "mic_obj": parsed_json.get("micro_objects", "")[:250],
            "mes_inf": parsed_json.get("meso_infrastructure", "")[:250],
            "mac_lu": parsed_json.get("macro_land_use", "")[:250],
            "spa_rel": parsed_json.get("spatial_relations", "")[:250],
            "hol_ctx": parsed_json.get("holistic_geographic_context", "")[:250]
        })

    # ==========================================
    # 步骤 5：生成并保存 Shapefile
    # ==========================================
    if gdf_records:
        gdf = gpd.GeoDataFrame(gdf_records, crs="EPSG:4326")
        out_shp_path = os.path.join(shp_dir, "SVI_Context_Results.shp")
        gdf.to_file(out_shp_path, driver='ESRI Shapefile', encoding='utf-8')
        print(f"\n✅ 成功！空间矢量文件已保存至: {out_shp_path}")
    else:
        print("\n❌ 警告：未能生成任何有效记录，Shapefile 创建失败。")

    return results_db