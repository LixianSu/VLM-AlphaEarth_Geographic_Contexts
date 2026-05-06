# =====================================================================
# 终极版 SVI 语境语义提取与空间矢量化管线 (资源受控版) Zero Shot
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

# 从你的配置中心导入静态路径（注意：确保 Config 里不再自动加载模型！）
from main_entry.config import Config


def setup_controlled_vlm():
    """
    按需分配 GPU 资源并加载模型，避免挤占实验室公共算力。
    :param gpu_ids: list, 指定使用的显卡编号，例如只用第一张卡 [0]，或双卡 [0, 1]
    :param vram_limit_gb: int, 每张卡允许使用的最大显存 (GB)
    """
    gpu_ids = Config.gpu_id

    vram_limit_gb = Config.vram_limit_gb

    print(f"🔧 正在配置计算沙箱: 启用 GPU {gpu_ids}, 单卡显存上限: {vram_limit_gb}GB")

    # 1. 物理隔离：对 PyTorch 隐藏未被选中的显卡 (必须在初始化 CUDA 前设置)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    # 2. 显存节流阀：告诉 Hugging Face 的 accelerate 库每张卡的显存上限
    # 构造 max_memory 字典，例如 {0: "16GiB"}
    max_mem_dict = {i: f"{vram_limit_gb}GiB" for i in range(len(gpu_ids))}

    print("⏳ 正在按设定的资源配额加载 Qwen2.5-VL 7B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem_dict  # 核心参数：强制限制显存使用量
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("✅ 模型受控加载完成！")

    return model, processor


def qwen25vl_to_shapefile(VLM_model, processor, gpu_device="cuda"):
    """
    读取指定文件夹下的街景，联合输出多尺度 Geographic Context，
    并自动将结果导出为包含空间坐标与丰富语义属性的 Shapefile。
    """
    # 路径依然从 Config 获取
    folder_path = Config.CLEANED_SVI_DIR

    all_images = glob.glob(os.path.join(folder_path, "*.jpg"))
    location_groups = {}

    # ==========================================
    # 步骤 1：解析文件并提取经纬度坐标
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
                location_groups[loc_id] = {'coords': (lon, lat), 'images': []}

            location_groups[loc_id]['images'].append((heading, img_path))

    results_db = {}
    gdf_records = []

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
        ).to(gpu_device)  # 将张量推送到正确的设备

        # 执行推理
        generated_ids = VLM_model.generate(**inputs, max_new_tokens=1024, temperature=0.0)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        # ==========================================
        # 步骤 3：JSON 防御性解析
        # ==========================================
        try:
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

        gdf_records.append({
            "loc_id": loc_id,
            "geometry": Point(lon, lat),
            "mic_obj": parsed_json.get("micro_objects", "")[:250],  # 防止超出Shapefile长度限制
            "mes_inf": parsed_json.get("meso_infrastructure", "")[:250],
            "mac_lu": parsed_json.get("macro_land_use", "")[:250],
            "spa_rel": parsed_json.get("spatial_relations", "")[:250],
            "hol_ctx": parsed_json.get("holistic_geographic_context", "")[:250]
        })

    # ==========================================
    # 步骤 4：生成并保存 Shapefile
    # ==========================================
    if gdf_records:
        shp_dir = os.path.join(folder_path, "shp")
        os.makedirs(shp_dir, exist_ok=True)
        gdf = gpd.GeoDataFrame(gdf_records, crs="EPSG:4326")
        out_shp_path = os.path.join(shp_dir, "SVI_Context_Results.shp")
        gdf.to_file(out_shp_path, driver='ESRI Shapefile', encoding='utf-8')
        print(f"\n✅ 成功！空间矢量文件已保存至: {out_shp_path}")
    else:
        print("\n❌ 警告：未能生成任何有效记录，Shapefile 创建失败。")

    return results_db

# -----------------
# 运行调用示例 (展示如何精确控制资源)
# -----------------
# if __name__ == "__main__":
#     # 场景 A：我只想用第 0 张卡，且最多给它 16GB 显存，把第 1 张卡留给同门师弟跑 YOLO
#     my_model, my_processor = setup_controlled_vlm(gpu_ids=[0], vram_limit_gb=16)
#
#     # 将受控的模型传入干活的管线
#     final_results = qwen25vl_to_shapefile(my_model, my_processor, gpu_device="cuda:0")