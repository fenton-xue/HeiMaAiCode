import torch
from torch.utils.data import TensorDataset, DataLoader # 构造数据集对象
from torch.utils.data import DataLoader # 数据加载器
from torch.nn import Linear, MSELoss # nn模块中有平方损失函数和假设函数
from torch.optim import SGD # optim模块中有优化器函数
from sklearn.datasets import make_regression # 创建线性回归模型数据集
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 1、数据集构建
def create_dataset():
    x, y, coef = make_regression(n_samples=100, # 样本个数100
                                 n_features=1,  # 特征向量的维度是1
                                 noise=10,      # 添加的噪声是10
                                 coef=True,     # 生成线性回归的数据权重是返回出来的
                                 bias=1.5,      # 偏置是1.5
                                 random_state=0) # 随机数种子是0
    # 将构建数据转换为张量类型
    x = torch.tensor(x)
    y = torch.tensor(y)

    return x, y, coef

if __name__ == '__main__':
    # 构造数据集
    x, y, coef = create_dataset()

    # 构造数据集对象
    dataset = TensorDataset(x, y)

    # 构造数据加载器
    # dataset=: 数据集对象
    # batch_size=:批量训练样本数据
    # shuffle=:样本数据是否进行乱序
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    # 构造模型
    # in_features指的是输入张量的大小size
    # out_features指的是输出张量的大小size
    model = Linear(in_features=1, out_features=1)

    # 设置损失函数和优化器
    # 构造平方损失函数
    loss = MSELoss()
    # 构造优化函数(参数是全部的数据，lr是学习率，0.001)
    optimizer = SGD(params=model.parameters(), lr=0.001)

    # 指定训练轮次为100
    epochs = 100
    # 损失的变化
    loss_epoch = [] # 记录损失每次的平均损失
    total_loss = 0.0 # 记录每个轮次的损失之和，也就是总损失
    train_sample = 0.0 # 记录训练样本个数
    for _ in range(epochs):
        for train_x, train_y in dataloader:
            # 将一个batch的训练数据送入模型
            y_pred = model(train_x.type(torch.float32))
            # 计算损失值
            loss_values = loss(y_pred, train_y.reshape(-1, 1).type(torch.float32))
            total_loss += loss_values
            train_sample += len(train_y)
            # 梯度清零
            optimizer.zero_grad()
            # 自动微分（反向传播）
            loss_values.backward()
            # 更新参数
            optimizer.step()
        # 获取每个batch的损失
        loss_epoch.append(total_loss.detach().numpy() / train_sample)

    # 绘制损失变化曲线
    plt.plot(range(epochs), loss_epoch)
    plt.title('损失变化曲线')
    # plt.grid()
    plt.show()

    # 绘制拟合直线
    plt.scatter(x, y)
    x = torch.linspace(x.min(), x.max(), 1000)
    y1 = torch.tensor([v * model.weight + model.bias for v in x])
    y2 = torch.tensor([v * coef + 1.5 for v in x])

    plt.plot(x, y1, label='训练')
    plt.plot(x, y2, label='真实')
    plt.grid()
    plt.legend()
    plt.show()