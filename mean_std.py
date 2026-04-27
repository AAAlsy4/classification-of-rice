from PIL import Image
import os
import numpy as np

# 文件夹路径，包含所有图片文件
folder_path = 'Rice_Image_Dataset'

# 初始化累积变量
total_pixels = 0  # 记录所有图像的总像素数
sum_normalized_pixel_values = np.zeros(3)  # 如果是RGB图像，需要三个通道的均值和方差

# 第一次遍历：计算均值
# 遍历文件夹中的图片文件
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        # 检查文件是否为图片格式
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 可根据实际情况添加其他格式
            # 构建完整的图片路径
            image_path = os.path.join(root, filename)
            # 使用PIL打开图片
            image = Image.open(image_path)
            # 将图片转换为numpy数组，形状为 (高度, 宽度, 通道数)
            image_array = np.array(image)

            # 归一化像素值到0-1之间（将0-255的像素值除以255）
            normalized_image_array = image_array / 255.0

            # 调试信息：打印图片路径和形状
            # print(image_path)
            # print(normalized_image_array.shape)

            # 累积归一化后的像素值和像素数量
            total_pixels += normalized_image_array.size  # 累加总像素数（高度×宽度×通道数）
            # 按通道求和，axis=(0,1)表示在高度和宽度维度上求和，保留通道维度
            sum_normalized_pixel_values += np.sum(normalized_image_array, axis=(0, 1))

# 计算均值：每个通道的总和除以总像素数
# 注意：这里total_pixels是所有通道的总像素数，所以每个通道的均值计算是正确的
mean = sum_normalized_pixel_values / total_pixels

# 第二次遍历：计算方差
# 初始化平方差累积变量
sum_squared_diff = np.zeros(3)

# 再次遍历所有图片文件
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            image_array = np.array(image)
            # 归一化像素值到0-1之间
            normalized_image_array = image_array / 255.0

            # 调试信息
            # print(normalized_image_array.shape)
            # print(mean.shape)
            # print(image_path)

            # 计算每个像素值与均值的平方差
            try:
                # 计算每个像素与对应通道均值的差值，然后平方
                diff = (normalized_image_array - mean) ** 2
                # 按通道求和平方差
                sum_squared_diff += np.sum(diff, axis=(0, 1))
            except Exception as e:
                # 异常处理：如果出现形状不匹配等问题
                print(f"捕获到异常: {e}，文件: {image_path}")

            # 原始代码（无异常处理）：
            # diff = (normalized_image_array - mean) ** 2
            # sum_squared_diff += np.sum(diff, axis=(0, 1))

# 计算方差：平方差的总和除以总像素数
variance = sum_squared_diff / total_pixels

# 输出结果
print("Mean:", mean)  # 每个通道的均值 [R_mean, G_mean, B_mean]
print("Variance:", variance)  # 每个通道的方差 [R_var, G_var, B_var]