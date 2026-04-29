import torch
import torch.utils.data as Data  # 数据加载工具
from torchvision import transforms  # 图像预处理变换
from model import ResNet18,Residual
from torchvision.datasets import ImageFolder # 图片加载
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse  # 命令行参数解析库
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


def test_data_process(data_dir):
    """
        测试集数据处理函数

        功能：
        1. 定义数据预处理变换
        2. 加载训练数据集
        3. 将数据集划分为训练集和验证集
        4. 创建训练和验证数据加载器

        返回：
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
    """

    # 数据集归一化
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    normalize = transforms.Normalize(mean=mean,std=std)

    # 定义数据预处理变换序列
    # transforms.Compose用于将多个数据预处理操作组合在一起
    test_transforms = transforms.Compose([
        # 调整图像大小为224x224像素
        # 这个尺寸是很多预训练模型的标准输入尺寸
        transforms.Resize((224, 224)),

        # 将PIL图像或numpy数组转换为PyTorch张量
        # 同时会自动将像素值从[0,255]范围缩放到[0.0,1.0]范围
        transforms.ToTensor(),

        # 标准化图像，使其归一化到[0.0,1.0]范围
        normalize
    ])

    # 使用ImageFolder加载数据集
    # ImageFolder会自动根据文件夹结构创建标签，每个子文件夹代表一个类别
    # root: 数据集根目录路径
    # transform: 应用于每个图像的数据预处理变换
    test_data = ImageFolder(root=data_dir, transform=test_transforms)

    # 创建测试数据加载器
    test_dataloader = Data.DataLoader(dataset=test_data,  # 测试数据集
                                      batch_size=1,  # 每批处理1个样本，便于逐个样本测试和可视化
                                      shuffle=True,  # 打乱数据顺序，确保测试的随机性
                                      num_workers=0)  # 不使用子进程加载数据（通常测试时数据量较小）

    return test_dataloader


def test_model_process(model, test_dataloader):
    """
    模型测试函数：在测试集上评估模型性能

    功能说明：
    1. 设置测试设备（GPU/CPU）
    2. 将模型移动到指定设备
    3. 在测试集上进行前向传播计算
    4. 统计模型在测试集上的准确率

    参数:
        model: 已经训练好的神经网络模型实例
        test_dataloader: 测试数据加载器，提供测试数据批次

    返回:
        无直接返回值，但会打印测试准确率
    """
    # 检测可用设备，优先使用GPU（CUDA），如果没有则使用CPU
    # 确保测试时使用的设备与训练时一致
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将模型移动到指定设备（GPU或CPU）
    model = model.to(device)

    # 初始化测试统计变量
    test_corrects = 0  # 测试正确预测数，统计测试集中正确分类的样本数量
    test_num = 0  # 测试样本总数，统计测试集总样本数量

    num = 1
    classes = ['Arborio','Basmati','Ipsala','Jasmine','Karacadag']
    inference_predict = []
    inference_real = []

    # 使用torch.no_grad()上下文管理器，禁用梯度计算
    # 在测试阶段不需要计算梯度，可以节省内存和计算资源
    with torch.no_grad():
        # 遍历测试数据加载器中的所有样本
        for test_data_x, test_data_y in tqdm(test_dataloader, total=len(test_dataloader)):
            # 将当前测试样本的输入数据和标签移动到指定设备
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 设置模型为评估模式（禁用dropout和batch normalization的训练行为）
            # 确保测试时模型行为与验证时一致
            model.eval()

            # 前向传播，得到模型预测输出（每个类别的得分/概率）
            output = model(test_data_x)

            # 获取预测标签（取最大概率的类别）
            # torch.argmax返回指定维度（dim=1表示类别维度）最大值的索引
            pre_lab = torch.argmax(output, dim=1)

            # 统计正确预测的数量，pre_lab == test_data_y.data比较预测标签和真实标签
            test_corrects += torch.sum(pre_lab == test_data_y.data)

            # 统计已处理的测试样本总数，用于计算准确率
            test_num += test_data_x.size(0)  # 这里batch_size=1，所以每次增加1

            # 测试集推理
            result = pre_lab.item()
            label = test_data_y.item()
            if num <= 10:
                num += 1
                inference_predict.append(classes[result])
                inference_real.append(classes[label])

    # 计算测试准确率：正确预测数 / 总样本数
    # .double()确保使用双精度计算，提高计算精度
    test_acc = test_corrects.double() / test_num

    # 打印测试准确率，格式化为4位小数
    print('Test Acc: {:.4f}'.format(test_acc))
    for i in range(len(inference_predict)):
        print('Predict:',inference_predict[i],' Real:',inference_real[i])


def parse_args():
    parser = argparse.ArgumentParser(description='Test ResNet')

    parser.add_argument('--data_dir', type=str, default='./data/test', help='Directory containing test data')
    parser.add_argument('--pth_file', type=str, default='./model_best.pth', help='Path to the trained model file')
    parser.add_argument('--inference', action='store_true', help='Whether to perform inference on a single image')
    parser.add_argument('--image_path', type=str, default='./data/train/Arborio/Arborio (1).jpg', help='Path to the image for inference')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 将模型实例化
    print('实例化模型')
    model = ResNet18(Residual)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('加载模型权重')
    model.load_state_dict(torch.load(args.pth_file, map_location=device))
    print('处理测试数据')
    test_dataloader = test_data_process(args.data_dir)

    model = model.to(device)

    print('开始测试')
    test_model_process(model, test_dataloader)

    # 测试单张图片
    if args.inference:
        print('单张图片推理')
        img = Image.open(args.image_path)
        mean = np.load('mean.npy')
        std = np.load('std.npy')
        normalize = transforms.Normalize(mean=mean, std=std)
        test_transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),normalize])
        img = test_transforms(img)
        img = img.to(device)

        # 将图片放入张量, 并增加一个维度，使其符合输入要求
        img = img.unsqueeze(0)
        with torch.no_grad():
            model.eval()
            output = model(img)
            pre_lab = torch.argmax(output, dim=1)
            classes = ['Arborio','Basmati','Ipsala','Jasmine','Karacadag']
            print(classes[pre_lab.item()])