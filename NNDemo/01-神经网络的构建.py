import torch
import torch.nn as nn

# 计算模型参数，查看模型结构
from torchsummary import summary

# 创建神经网络模型类
class Model(nn.Module):
    # 初始化属性值
    def __init__(self):
        # 调用父类的初始化属性值
        super(Model, self).__init__()
        # 创建第一个隐藏层模型，3个输入特征，3个输出特征
        self.linear1 = nn.Linear(3, 3)
        # 创建第二个隐藏层模型，3个输入特征，2个输出特征
        self.linear2 = nn.Linear(3, 2)
        # 创建输出层模型
        self.out = nn.Linear(2, 2)

    # 创建前向传播方法，自动执行forward()
    def forward(self, x):
        # 数据经过第一个线性层
        x = self.linear1(x)
        # 使用sigmoid激活函数（为了演示，一般隐藏层不用sigmoid）
        x = torch.sigmoid(x)
        # 数据经过第二个线性层
        x = self.linear2(x)
        # 使用ReLU激活函数
        x = torch.relu(x)
        # 数据经过输出层
        x = self.out(x)
        # 使用softmax激活函数
        # 在dim=-1维度数据相加为1
        x = torch.softmax(x, dim=-1)
        return x

if __name__ == '__main__':
    # 实例化model对象
    my_model = Model()
    # 随机产生数据
    my_data = torch.randn(5, 3)
    print("mydata shape:", my_data.shape)
    # 数据经过神经网络模型训练
    output = my_model(my_data)
    print("output shape:", output.shape)
    # 计算模型参数
    # 计算每层每个神经元的w和b个数总和
    summary(my_model, input_size=(3,), batch_size=5)
