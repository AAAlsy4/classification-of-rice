from PIL import Image
import os
import numpy as np
from tqdm import tqdm

folder_path = 'Rice_Image_Dataset'

# 收集图片路径
image_paths = []
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_paths.append(os.path.join(root, filename))

# 第一次遍历：计算均值
total_spatial_pixels = 0           # 只记空间像素数（H*W）
sum_pixel_values = np.zeros(3)

for impath in tqdm(image_paths, total=len(image_paths), desc='计算均值'):
    img = Image.open(impath).convert('RGB')   # 确保 RGB
    arr = np.array(img) / 255.0
    h, w, _ = arr.shape
    total_spatial_pixels += h * w
    sum_pixel_values += np.sum(arr, axis=(0, 1))

mean = sum_pixel_values / total_spatial_pixels

# 第二次遍历：计算方差
sum_squared_diff = np.zeros(3)

for impath in tqdm(image_paths, desc='计算方差'):
    img = Image.open(impath).convert('RGB')
    arr = np.array(img) / 255.0
    diff = (arr - mean) ** 2
    sum_squared_diff += np.sum(diff, axis=(0, 1))

variance = sum_squared_diff / total_spatial_pixels

print("Mean (R, G, B):", mean)
print("Variance (R, G, B):", variance)

np.save('mean.npy', mean)
np.save('std.npy', variance)