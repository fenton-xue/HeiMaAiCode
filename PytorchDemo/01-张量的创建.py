import torch
import numpy as np

# # 1、创建张量标量
# data = torch.tensor(10)
# print(data)
#
# # 2、 numpy 数组，由于 data 为 float64， 下面代码也使用该类型
# # 创建一个2 * 3 的随机数组
# data = np.random.randn(2, 3)
# data = torch.tensor(data)
# print(data)
#
# # 创建2行3列的张量，默认dtype为float32
# data = torch.Tensor(2, 3)
# print(data)
#
# data = torch.Tensor([10])
# print(data)
#
# data = torch.Tensor([10, 20])
# print(data)
#
# # 3、列表，下面代码使用默认元素类型 float32
# data = [[10., 20., 30.], [40., 50., 60.]]
# data = torch.tensor(data)
# print(data)
#
# # 1、在指定区间按照步长生成元素 [start, end, step)
# data = torch.arange(0, 10, 2)
# print(data)
#
# # 2、在指定区间按照元素个数生成 [start, end, num]
# data = torch.linspace(0, 11, 10)
# print(data)
#
# # 1、 创建随机张量
# data = torch.randn(2, 3)
# print(data)
#
# # 1、创建指定形状全0张量
# data = torch.zeros(2, 3)
# print(data)
#
# # 2、创建指定形状全1张量
# data = torch.ones(2, 3)
# print(data)
#
# 3、创建指定形状指定值的张量
# data = torch.full([2, 3], 10)
# print(data)

# data = torch.full([2, 3], 10)
# print(data.dtype)
# 将 data 元素类型转换为 float64 类型
# data = data.type(torch.DoubleTensor)
# print(data.dtype)
# 转换为其他类型
# data = data.type(torch.IntTensor)
# data = data.type(torch.LongTensor)
# data = data.type(torch.FloatTensor)

data = torch.full([2, 3], 10)
print(data.dtype)

# 将 data 元素类型转换为 float64 类型
data = data.double()
print(data.dtype)
# 转换为其他类型
# data = data.int()
# data = data.long()
# data = data.float()

