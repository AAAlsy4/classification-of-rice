import copy  # 用于深度复制模型权重
import torch
from torch import nn
from torchvision import transforms  # 图像预处理变换
from torchvision.datasets import ImageFolder # 图片加载
import torch.utils.data as Data  # 数据加载工具
import matplotlib.pyplot as plt
from model import ResNet18,Residual,ViT
import time  # 时间模块，用于计算训练时间
import pandas as pd  # 数据处理库，用于保存训练过程数据
import numpy as np
from tqdm import tqdm
import argparse  # 命令行参数解析库
from torch.utils.tensorboard import SummaryWriter


def train_val_data_process(batch_size, num_workers, data_dir):
    """
        训练集和验证集数据处理函数

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
    normalize = transforms.Normalize(mean=mean, std=std)


    # 定义数据预处理变换序列
    # transforms.Compose用于将多个数据预处理操作组合在一起
    train_transforms = transforms.Compose([
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
    train_data = ImageFolder(root=data_dir, transform=train_transforms)

    # 将训练数据集按8:2的比例随机划分为训练集和验证集
    # random_split函数将数据集随机分割成指定大小的子集
    train_data, val_data = Data.random_split(train_data,
                                             [round(0.8 * len(train_data)),  # 80%作为训练集，round用于四舍五入取整
                                              round(0.2 * len(train_data))])  # 20%作为验证集

    # 创建训练数据加载器
    train_dataloader = Data.DataLoader(dataset=train_data,  # 训练数据集
                                       batch_size=batch_size,  # 每批处理batch_size个样本，批量大小影响训练速度和内存使用
                                       shuffle=True,  # 打乱数据顺序，防止模型学习到数据顺序特征
                                       num_workers=num_workers)  # 使用num_workers个子进程加载数据，加速数据读取

    # 创建验证数据加载器
    val_dataloader = Data.DataLoader(dataset=val_data,  # 验证数据集
                                     batch_size=batch_size,  # 每批处理batch_size个样本，与训练集保持一致
                                     shuffle=True,  # 打乱数据顺序，确保评估的随机性
                                     num_workers=num_workers)  # 使用num_workers个子进程加载数据

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs, learning_rate):
    """
    模型训练和验证函数

    功能说明：
    1. 设置训练设备（GPU/CPU）
    2. 定义优化器和损失函数
    3. 执行多轮训练和验证循环
    4. 记录训练过程中的损失和准确率
    5. 保存最佳模型权重

    参数:
        model: 要训练的神经网络模型实例（如LeNet）
        train_dataloader: 训练数据加载器，提供训练数据批次
        val_dataloader: 验证数据加载器，提供验证数据批次
        num_epochs: 训练的总轮数，每个epoch代表完整遍历一次训练集

    返回:
        train_process: DataFrame包含训练过程中的损失和准确率记录
    """
    # 检测可用设备，优先使用GPU（CUDA），如果没有则使用CPU
    # torch.cuda.is_available()检查系统是否支持CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义优化器：使用Adam优化算法，学习率为0.001
    # Adam优化器结合了动量法和自适应学习率的优点，适合大多数深度学习任务
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 定义损失函数：交叉熵损失，适用于多分类问题
    # 交叉熵损失函数衡量模型预测概率分布与真实标签分布的差异
    criterion = nn.CrossEntropyLoss()

    # 将模型移动到指定设备（GPU或CPU）
    # 如果使用GPU，可以显著加速模型计算
    model = model.to(device)

    writer = SummaryWriter(log_dir=f'./tensorboard_{args.model}')  # 创建TensorBoard SummaryWriter实例，用于记录训练过程数据

    # 深度复制当前模型权重，用于保存最佳模型
    # copy.deepcopy确保完全独立复制，避免后续修改影响原始状态
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # 初始化最佳准确率，用于跟踪训练过程中最好的验证集准确率

    # 初始化列表用于记录训练和验证过程中的损失和准确率
    train_loss_list = []  # 训练损失列表，记录每个epoch的平均训练损失
    val_loss_list = []  # 验证损失列表，记录每个epoch的平均验证损失
    train_acc_list = []  # 训练准确率列表，记录每个epoch的训练准确率
    val_acc_list = []  # 验证准确率列表，记录每个epoch的验证准确率

    since = time.time()  # 记录训练开始时间，用于计算总训练时长

    # 开始训练循环，遍历每个epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))  # 打印当前epoch信息，格式化为"当前epoch/总epochs"
        print('-' * 10)  # 打印分隔线，便于阅读输出

        # 初始化每个epoch的统计变量
        train_loss = 0.0  # 训练损失累计值，用于计算该epoch的平均训练损失
        train_corrects = 0  # 训练正确预测数，统计该epoch中正确分类的样本数量
        val_loss = 0.0  # 验证损失累计值，用于计算该epoch的平均验证损失
        val_corrects = 0  # 验证正确预测数，统计验证集中正确分类的样本数量
        train_num = 0  # 训练样本总数，统计该epoch处理的训练样本数量
        val_num = 0  # 验证样本总数，统计该epoch处理的验证样本数量

        # 训练阶段：遍历训练数据加载器中的所有批次
        # enumerate提供批次索引和批次数据，便于调试和监控
        for step, (b_x, b_y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # 将当前批次的输入数据(b_x)和标签(b_y)移动到指定设备（GPU/CPU）
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()  # 设置模型为训练模式（启用dropout和batch normalization）
            # 训练模式下，模型会启用所有的训练特定层，如dropout和batch normalization的训练行为

            output = model(b_x)  # 前向传播，得到模型预测输出（每个类别的得分/概率）
            # output的形状通常为[batch_size, num_classes]，表示每个样本属于各个类别的预测得分

            pre_lab = torch.argmax(output, dim=1)  # 获取预测标签（取最大概率的类别）
            # torch.argmax返回指定维度（dim=1表示类别维度）最大值的索引，即预测的类别标签

            loss = criterion(output, b_y)  # 计算损失，比较模型输出和真实标签

            # 反向传播过程（梯度计算和参数更新）
            optimizer.zero_grad()  # 清空梯度缓存，防止梯度累加
            loss.backward()  # 反向传播计算梯度，通过链式法则计算各参数梯度
            optimizer.step()  # 更新模型参数，根据梯度调整权重

            # 累计训练损失（乘以batch_size是因为损失是batch的平均值，需要还原为总和）
            train_loss += loss.item() * b_x.size(0)
            writer.add_scalar('Loss/train', loss.item(), epoch)  # 将当前批次的训练损失记录到TensorBoard
            
            # loss.item()获取标量损失值，b_x.size(0)是当前批次的样本数量

            # 统计正确预测的数量，pre_lab == b_y.data比较预测标签和真实标签
            train_corrects += torch.sum(pre_lab == b_y.data)
            
            # 统计已处理的训练样本总数，用于计算平均损失和准确率
            train_num += b_x.size(0)
            writer.add_scalar('Acc/train', train_corrects.double().item() / train_num, epoch) # 将当前批次的训练准确率记录到TensorBoard

        # 验证阶段：遍历验证数据加载器中的所有批次
        # 验证阶段不更新模型参数，只进行前向传播计算损失和准确率
        for step, (b_x, b_y) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            # 将验证数据移动到指定设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()  # 设置模型为评估模式（禁用dropout和batch normalization）
            # 评估模式下，模型会使用固定的统计量进行batch normalization，并关闭dropout

            output = model(b_x)  # 前向传播，得到模型预测输出
            pre_lab = torch.argmax(output, dim=1)  # 获取预测标签
            loss = criterion(output, b_y)  # 计算损失

            # 累计验证损失
            val_loss += loss.item() * b_x.size(0)
            writer.add_scalar('Loss/val', loss.item(), epoch)  # 将当前批次的验证损失记录到TensorBoard
            # 统计正确预测的数量
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 统计已处理的验证样本总数
            val_num += b_x.size(0)
            writer.add_scalar('Acc/val', val_corrects.double().item() / val_num, epoch) # 将当前批次的验证准确率记录到TensorBoard

        # 计算并记录当前epoch的平均训练损失和准确率
        train_loss_list.append(train_loss / train_num)  # 平均训练损失 = 总损失 / 样本数
        train_acc_list.append(train_corrects.double().item() / train_num)  # 训练准确率 = 正确数 / 总数
        # .double()确保使用双精度计算，.item()将张量转换为Python数值

        # 计算并记录当前epoch的平均验证损失和准确率
        val_loss_list.append(val_loss / val_num)  # 平均验证损失
        val_acc_list.append(val_corrects.double().item() / val_num)  # 验证准确率

        # 打印当前epoch的训练和验证结果
        print('{} Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch + 1, train_loss_list[-1], train_acc_list[-1]))
        print('{} Val Loss:{:.4f} Val Acc:{:.4f}'.format(epoch + 1, val_loss_list[-1], val_acc_list[-1]))

        # 更新最佳模型：如果当前epoch的验证准确率高于历史最佳，则保存当前模型
        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]  # 更新最佳准确率
            best_model_wts = copy.deepcopy(model.state_dict())  # 深度复制当前模型权重
            print('Best model updated at epoch {} with val acc: {:.4f}'.format(epoch + 1, best_acc))  # 打印更新最佳模型的信息

        # 计算并打印已用时间
        time_use = time.time() - since  # 计算从训练开始到现在的时间差
        print('Time use:{:.0f}m {:.0f}s'.format(time_use // 60, time_use % 60))  # 格式化为分钟和秒

    # 保存最佳模型权重（验证集上表现最好的模型）到文件，便于后续使用或部署
    torch.save(best_model_wts, f'model_best_{args.model}.pth')
    print(f'Best model saved to ./model_best_{args.model}.pth')  # 打印保存最佳模型的提示信息

    print('Best Val Acc: {:4f}'.format(best_acc))  # 打印最佳验证准确率

    # 将训练过程数据整理为DataFrame，便于分析和可视化
    train_process = pd.DataFrame(data={'epoch': range(num_epochs),  # epoch编号
                                       'train_loss': train_loss_list,  # 训练损失记录
                                       'train_acc': train_acc_list,  # 训练准确率记录
                                       'val_loss': val_loss_list,  # 验证损失记录
                                       'val_acc': val_acc_list})  # 验证准确率记录

    return train_process  # 返回训练过程数据


def matplot_acc_loss(train_process):
    """
    训练过程可视化函数：绘制损失和准确率随epoch变化的曲线

    功能说明：
    1. 创建包含两个子图的图形窗口
    2. 左子图显示训练损失和验证损失曲线
    3. 右子图显示训练准确率和验证准确率曲线
    4. 添加图例、坐标轴标签等图形元素

    参数:
        train_process: DataFrame包含训练过程中的损失和准确率记录
    """
    plt.figure(figsize=(12, 4))  # 创建图形窗口，设置大小为12x4英寸

    # 第一个子图：损失曲线
    plt.subplot(1, 2, 1)  # 创建1行2列的子图布局，当前为第1个子图
    plt.plot(train_process['epoch'], train_process['train_loss'], 'ro', label='Train Loss')  # 训练损失曲线，红色圆点
    plt.plot(train_process['epoch'], train_process['val_loss'], 'bs-', label='Val Loss')  # 验证损失曲线，蓝色实线方块
    plt.legend()  # 显示图例
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Loss')  # y轴标签

    # 第二个子图：准确率曲线
    plt.subplot(1, 2, 2)  # 当前为第2个子图
    plt.plot(train_process['epoch'], train_process['train_acc'], 'ro-', label='Train Acc')  # 训练准确率曲线，红色实线圆点
    plt.plot(train_process['epoch'], train_process['val_acc'], 'bs-', label='Val Acc')  # 验证准确率曲线，蓝色实线方块
    plt.legend()  # 显示图例
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Acc')  # y轴标签
    plt.savefig(f'train_process_{args.model}.jpg')  # 保存图形到文件，便于后续查看
    print(f'saved to ./train_process_{args.model}.jpg')  # 打印提示信息
    plt.show()  # 显示图形

def parse_args():
    parser = argparse.ArgumentParser(description="Train")

    parser.add_argument('--model', type=str, default='ResNet', choices=['ResNet', 'ViT'], help='Model architecture to train (default: ResNet)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--data_dir', type=str, default='./data/train', help='Directory containing training data')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize training process')

    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 将模型实例化
    print(f'实例化模型: {args.model}')
    if args.model == 'ResNet':
        model = ResNet18(Residual)
    else:
        model = ViT()
    print('数据处理')
    train_dataloader, val_dataloader = train_val_data_process(args.batch_size, args.num_workers, args.data_dir)
    print('开始训练')
    train_process = train_model_process(model, train_dataloader, val_dataloader, args.epochs, args.learning_rate)
    if args.visualize:
        print('可视化训练过程')
        matplot_acc_loss(train_process)