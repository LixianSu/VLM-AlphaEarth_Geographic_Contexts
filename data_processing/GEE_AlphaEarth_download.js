// 1. 定义香港精确的矩形包围盒
var hk_bounds = ee.Geometry.Rectangle([113.83, 22.13, 114.45, 22.57]);

// 2. 加载 2020 年 AlphaEarth 数据 (遵循官方文档示例)
var aef_2020 = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                 .filterDate('2020-01-01', '2021-01-01')
                 .filterBounds(hk_bounds)
                 .mosaic() // 应对 163km 瓦片切分问题，无缝缝合
                 .clip(hk_bounds);

// 3. 准备导出影像
// 文档标明取值为 -1 到 1，Float32 精度绝对够用，且能防超 16GB 的报错
var export_img = aef_2020.toFloat();

// 4. 执行导出任务
Export.image.toDrive({
  image: export_img,
  description: 'AEF_HK_2020_Official_Optimized',
  folder: 'AlphaEarth_Final',
  fileNamePrefix: 'AEF_HK_2020_Final',
  region: hk_bounds,

  // 完美契合文档中“本地通用横轴墨卡托投影生成”及“10米像素大小”的设定
  crs: 'EPSG:32650', // 香港 UTM 50N
  scale: 10,

  maxPixels: 1e13,
  fileDimensions: 7168, // 强行塞入单个文件
  formatOptions: {
    cloudOptimized: true
  }
});