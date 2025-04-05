import rasterio
import numpy as np
from pyproj import CRS
import math

def calculate_geotiff_area(filename, band=1, nodata=None, mask_condition=None):
    """
    计算GeoTIFF文件中有效区域的面积（平方米）。
    
    参数：
    filename: 输入的GeoTIFF文件路径
    band: 要读取的波段（默认为1）
    nodata: 手动指定nodata值（如果文件未正确设置）
    mask_condition: 自定义函数，用于生成有效区域掩膜，例如 lambda data: data > 0
    
    返回：
    有效区域的面积（平方米）
    """
    with rasterio.open(filename) as dataset:
        data = dataset.read(band)
        transform = dataset.transform
        crs = CRS.from_user_input(dataset.crs)
        
        # 处理nodata值
        if nodata is not None:
            mask = data != nodata
        elif dataset.nodata is not None:
            mask = data != dataset.nodata
        else:
            mask = np.ones(data.shape, dtype=bool)
        
        # 应用自定义掩膜条件
        if mask_condition is not None:
            mask &= mask_condition(data)
        
        # 检查坐标系类型
        if crs.is_geographic:
            # 地理坐标系（WGS84等）
            if transform.b != 0 or transform.d != 0:
                raise NotImplementedError("旋转的地理坐标系暂不支持")
            return _calculate_geographic_area(data, mask, transform)
        
        elif crs.is_projected:
            # 投影坐标系（UTM等）
            return _calculate_projected_area(mask, transform)
        
        else:
            raise ValueError("不支持的坐标系类型")

def _calculate_projected_area(mask, transform):
    """计算投影坐标系下的面积"""
    if transform.b != 0 or transform.d != 0:
        raise NotImplementedError("旋转的投影坐标系暂不支持")
    pixel_area = abs(transform.a * transform.e)  # 单位：平方米
    return np.count_nonzero(mask) * pixel_area

def _calculate_geographic_area(data, mask, transform):
    """计算地理坐标系下的面积（近似）"""
    height, width = data.shape
    total_area = 0.0
    m_per_deg = 111194.444  # 平均每度对应的米数
    
    for i in range(height):
        # 计算当前行的中心纬度
        _, y = transform * (0.5, i + 0.5)  # (列0.5, 行i+0.5)
        lat_rad = math.radians(y)
        
        # 计算像素尺寸（米）
        dx_deg = abs(transform.a)
        dy_deg = abs(transform.e)
        dx_m = dx_deg * m_per_deg * math.cos(lat_rad)
        dy_m = dy_deg * m_per_deg
        
        # 当前行有效像素数
        valid_pixels = np.count_nonzero(mask[i, :])
        total_area += valid_pixels * dx_m * dy_m
    
    return total_area

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python count_area.py <输入文件>")
        sys.exit(1)
    

    area = calculate_geotiff_area(sys.argv[1])
    print(f"总面积: {area:.2f} 平方米")
    print(f"约 {area/1e6:.2f} 平方公里")
    print(f"约 {area/1e4:.2f} 公顷")