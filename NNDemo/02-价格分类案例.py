import torch
from torch.special import xlogy
# 数据处理的工具包
from torch.utils.data import TensorDataset, DataLoader
# 模型构建需要的包
import torch.nn as nn
# 进行优化的包
import torch.optim as optim

# 进行数据集划分
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 画图
import matplotlib.pyplot as plt
import numpy as np

# 数据在csv文件中
import pandas as pd
import time

from torchsummary import summary


# 构建数据集
def create_dataset():
    # 使用pandas读取数据
    data = pd.read_csv('data/手机价格预测.csv')
    # 特征值和目标值
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # 类型转换：特征值，目标值
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    # 数据集划分
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88)
    # 构建数据集，转换为pytorch的形式
    train_dataset = TensorDataset(torch.from_numpy(x_train.values), torch.tensor(y_train.values))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid.values), torch.tensor(y_valid.values))

    return train_dataset, valid_dataset, x_train.shape[1], len(np.unique(y))

# 构建网络模型
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PhonePriceModel, self).__init__()
        # 1、第一层，输入维度为20，输出维度为128
        self.linear1 = nn.Linear(input_dim, 128)
        # 2、第二层，输入维度为128，输出维度为256
        self.linear2 = nn.Linear(128, 256)
        # 3、第三层，输入维度为256，输出维度为4
        self.linear3 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 前向传播过程
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        output = self.linear3(x)
        return output

# 模型训练过程
def train(train_dataset, input_dim, class_num,):
    # 初始化模型
    model = PhonePriceModel(input_dim, class_num)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # 训练轮次
    num_epoch = 50

    # 遍历每个轮次的数据
    for epoch_idx in range(num_epoch):
        # 初始化数据加载器
        dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        # 训练时间
        start = time.time()
        # 计算损失
        total_loss = 0.0
        total_num = 1
        # 便利每个batch数据进行处理
        for x, y in dataloader:
            # 将数据送入网络中进行预测
            output = model(x)
            # 计算损失
            loss = criterion(output, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 损失计算
            total_num += 1
            total_loss += loss.item()
        # 打印损失变换结果
        print('epoch:%4s loss:%.2f, time:%.2fs' % (epoch_idx + 1, total_loss / total_num, time.time() - start))
    # 模型保存
    torch.save(model.state_dict(), 'model/phone4.pth')

def test(valid_dataset, input_dim, class_num):
    # 加载模型和训练好的网络参数
    model = PhonePriceModel(input_dim, class_num)
    model.load_state_dict(torch.load('model/phone4.pth'))

    # 构建加载器
    dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    # 评估测试集
    correct = 0
    # 便利测试集 中的数据
    for x, y in dataloader:
        # 将其送入网络中
        output = model(x)
        # 获取类别结果
        y_pred = torch.argmax(output, dim=1)
        # 获取预测正确的个数
        correct += (y_pred==y).sum()
    # 求预测精度
    print('Acc：%.5f'%(correct.item() / len(valid_dataset)))

if __name__ == '__main__':
    # 获取数据
    train_dataset, valid_dataset, input_dim, class_num = create_dataset()
    # print("输入特征数：", input_dim)
    # print("分类个数：", class_num)

    # 模型实例化
    # model = PhonePriceModel(input_dim, class_num)
    # summary(model, input_size=(input_dim,), batch_size=16)

    # 模型训练过程
    train(train_dataset, input_dim, class_num)
    test(train_dataset, input_dim, class_num)


