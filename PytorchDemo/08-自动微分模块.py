import torch

# 1、当x为标量时梯度的计算
def test01():
    x = torch.tensor(5)
    # 目标值
    y = torch.tensor(0.)

    # 设置要更新的权重和偏置的初始值
    w = torch.tensor(1., requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3., requires_grad=True, dtype=torch.float32)
    # requires_grad设置为True时，自动微分模块才可以对它进行梯度计算

    # 设置网络的输出值
    z = x * w + b

    # 设置损失函数，并进行损失的计算
    loss = torch.nn.MSELoss()
    loss = loss(z, y)

    # 自动微分
    loss.backward()

    # 打印w，b变量的梯度
    # backward函数计算的梯度值会存储在张量的grad变量中
    print("w的梯度：", w.grad)
    print("b的梯度：", b.grad)

def test02():
    # 输入张量 2 * 5
    x = torch.ones(2, 5)
    # 目标值
    y = torch.zeros(2, 3)

    # 设置需要更新的权重和偏置的初始值
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    # 设置网络的输出值
    # 标量的时候可以直接用 *
    z = torch.matmul(x, w) + b

    # 设置损失函数，并进行损失的计算
    loss = torch.nn.MSELoss()
    loss = loss(z, y)

    # 自动微分
    loss.backward()

    print("w的梯度：", w.grad)
    print("b的梯度：", b.grad)

if __name__ == '__main__':
    # test01()
    test02()
    pass