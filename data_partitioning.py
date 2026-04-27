import os
from shutil import copy
import random


def mkfile(file):
    """
    创建目录的函数
    如果目录不存在则创建，存在则不执行任何操作

    参数:
        file: 要创建的目录路径
    """
    if not os.path.exists(file):
        os.makedirs(file)


# 原始数据集路径，包含猫和狗两个类别的图片
file_path = 'Rice_Image_Dataset'

# 获取data文件夹下所有文件夹名（即需要分类的类名）
# 例如：['cat', 'dog']
flower_class = [cla for cla in os.listdir(file_path)]

# 创建训练集train文件夹，并由类名在其目录下创建子目录
mkfile('data/train')  # 创建主训练目录
for cla in flower_class:
    mkfile('data/train/' + cla)  # 为每个类别创建子目录，如：data/train/cat, data/train/dog

# 创建验证集val文件夹，并由类名在其目录下创建子目录
mkfile('data/test')  # 创建主测试目录
for cla in flower_class:
    mkfile('data/test/' + cla)  # 为每个类别创建子目录，如：data/test/cat, data/test/dog

# 划分比例，训练集 : 测试集 = 9 : 1
# 这意味着90%的数据用于训练，10%的数据用于测试
split_rate = 0.1

# 遍历所有类别的全部图像并按比例分成训练集和测试集
for cla in flower_class:
    # 构建当前类别的完整路径，例如：'data_cat_dog/cat/'
    cla_path = file_path + '/' + cla + '/'

    # 获取该类别目录下所有图像文件的名称列表
    images = os.listdir(cla_path)

    # 获取该类别图像的总数量
    num = len(images)

    # 从当前类别的所有图像中随机抽取指定比例作为测试集
    # random.sample() 函数用于无放回抽样，确保不会重复选择
    eval_index = random.sample(images, k=int(num * split_rate))

    # 遍历当前类别的每一张图像
    for index, image in enumerate(images):
        # 如果当前图像在测试集抽样列表中
        if image in eval_index:
            # 构建原始图像的完整路径
            image_path = cla_path + image
            # 构建目标路径（测试集对应类别的目录）
            new_path = 'data/test/' + cla
            # 将图像复制到测试集目录
            copy(image_path, new_path)
        # 如果当前图像不在测试集抽样列表中，则放入训练集
        else:
            # 构建原始图像的完整路径
            image_path = cla_path + image
            # 构建目标路径（训练集对应类别的目录）
            new_path = 'data/train/' + cla
            # 将图像复制到训练集目录
            copy(image_path, new_path)

        # 显示处理进度条
        # \r 表示回到行首，实现动态更新效果
        # end="" 防止换行，保持在同一行显示
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")

    # 每个类别处理完成后换行
    print()

# 所有处理完成提示
print("processing done!")