import numpy as np
from PIL import Image
import cv2


H, W = 480, 640

# 固定随机种子，方便复现实验
np.random.seed(42)

# =========================
# 1. 生成俯视 RGB 图：桌面 + 多个物体
# =========================

# 先在 BGR 空间构图，最后再转成 RGB 给 PIL
rgb_bgr = np.zeros((H, W, 3), dtype=np.uint8)

# 整个画面都是桌面（俯视），浅棕色
table_color_bgr = (150, 190, 200)  # 大致偏木纹色
rgb_bgr[:] = table_color_bgr

# 轻微纹理/噪声，让桌面不那么“纯色”
noise = np.random.normal(0, 4, size=(H, W, 3))
rgb_bgr = np.clip(rgb_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# 多个物体（矩形 + 圆形），颜色各不相同
num_objects = 6
colors_bgr = [
    (40, 40, 200),   # 红色系
    (40, 180, 40),   # 绿色系
    (200, 200, 40),  # 黄色系
    (200, 80, 40),   # 橙色系
    (180, 40, 180),  # 紫色系
    (40, 180, 180),  # 青色系
]

# 先生成一个“物体列表”，后面 RGB 和 depth 都复用，保证位置和形状一一对应
objects = []
for i in range(num_objects):
    shape_type = "rect" if np.random.rand() < 0.6 else "circle"
    color = colors_bgr[i % len(colors_bgr)]

    # 物体大致尺寸 & 位置（避免太靠边）
    min_size, max_size = 40, 100
    h_size = np.random.randint(min_size, max_size)
    w_size = np.random.randint(min_size, max_size)
    margin = 40
    cx = np.random.randint(margin, W - margin)
    cy = np.random.randint(margin, H - margin)

    # 随机指定该物体的“高度” -> 对应更近的深度（毫米）
    # 桌面 ~ 1000mm，相机在正上方，物体最高比桌面近 ~150mm
    z_obj = 1000 - np.random.randint(30, 150)

    objects.append(
        dict(
            shape_type=shape_type,
            color=color,
            h_size=h_size,
            w_size=w_size,
            cx=cx,
            cy=cy,
            z_obj=z_obj,
        )
    )

# 根据 objects 渲染 RGB
for obj in objects:
    shape_type = obj["shape_type"]
    color = obj["color"]
    h_size = obj["h_size"]
    w_size = obj["w_size"]
    cx = obj["cx"]
    cy = obj["cy"]

    if shape_type == "rect":
        x1 = max(cx - w_size // 2, 0)
        x2 = min(cx + w_size // 2, W - 1)
        y1 = max(cy - h_size // 2, 0)
        y2 = min(cy + h_size // 2, H - 1)
        cv2.rectangle(rgb_bgr, (x1, y1), (x2, y2), color, thickness=-1)
    else:
        radius = min(h_size, w_size) // 2
        cv2.circle(rgb_bgr, (cx, cy), radius, color, thickness=-1)

# 转成 RGB 存盘
rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
Image.fromarray(rgb).save("rgb.png")
print("Saved rgb.png (top-down table with multiple objects)")


# =========================
# 2. 生成对应深度图 (16-bit，单位：毫米)
# =========================

depth = np.zeros((H, W), dtype=np.uint16)

Z_TABLE = 1000  # 桌面距离相机约 1.0m（俯视）
depth[:] = Z_TABLE

# 使用同一份 objects，给对应区域赋更近的深度
for obj in objects:
    shape_type = obj["shape_type"]
    h_size = obj["h_size"]
    w_size = obj["w_size"]
    cx = obj["cx"]
    cy = obj["cy"]
    z_obj = obj["z_obj"]

    if shape_type == "rect":
        x1 = max(cx - w_size // 2, 0)
        x2 = min(cx + w_size // 2, W - 1)
        y1 = max(cy - h_size // 2, 0)
        y2 = min(cy + h_size // 2, H - 1)
        depth[y1:y2 + 1, x1:x2 + 1] = np.minimum(
            depth[y1:y2 + 1, x1:x2 + 1], np.uint16(z_obj)
        )
    else:
        radius = min(h_size, w_size) // 2
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        depth[mask] = np.minimum(depth[mask], np.uint16(z_obj))

# 深度图保存为 16-bit PNG（使用 OpenCV，Pillow 某些版本不支持 I;16 PNG）
cv2.imwrite("depth.png", depth)
print("Saved depth.png (16-bit PNG by OpenCV, top-down)")

print("俯视桌面多物体示例 RGBD 已生成：rgb.png 和 depth.png")