# classification-of-rice

基于 ResNet18 骨干网络的图像分类模型，在 [Rice-Image-Dataset](https://pan.baidu.com/s/1hUcjG1Z34-1FHOy9Y1mX4w?pwd=42s2) 数据集上训练，提供预训练权重[model_best.pth](https://pan.baidu.com/s/1s6ZwKulsGCNOa77650rhcg?pwd=yewf)。

## 数据集介绍

五类大米品种（Arborio、Basmati、Ipsala、Jasmine、Karacadag）

![Arborio](./figs/Arborio%20%281%29.jpg)

![Basmati](./figs/Basmati%20%281%29.jpg)

![Ipsala](./figs/Ipsala%20%28638%29.jpg)

![Jasmine](./figs/Jasmine%20%2810453%29.jpg)

![Karacadag](./figs/Karacadag%20%281158%29.jpg)

## 测试结果

![训练过程损失与准确率曲线](./figs/train_process.jpg)

实际准确率在99.0%-99.5%之间

## 使用

### 环境配置

```bash
conda env create -f environment.yml
```

### 数据集预处理

```bash
python data_partitioning.py # 划分数据集
```

```bash
python mean_std.py # 计算均值和方差，并保存
```

### 训练

```bash
python model_train.py --epochs 20 --batch_size 128 --num_workers 8 --learning_rate 0.001 --data_dir ./data/train --visualize
```

### 推理

```bash
python model_test.py --data_dir ./data/test --pth_file ./model_best.pth --inference --image_path ./Arborio.jpg
```

### 使用tensorboard

```bash
tensorboard --logdir ./tensorboard
```
