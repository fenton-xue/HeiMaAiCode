import torch
import numpy as np

# data = torch.randint(0, 10, [2, 3])
# print(data)
#
# # 1、不修改原数据
# new_data = data.add(10)
# print(new_data)
#
# # 2、直接修改原数据
# data.add_(10)
# print(data)
#
# print(data.sub(100))
# print(data.mul(100))
# print(data.div(100))
# print(data.neg())

# data1 = torch.tensor([[1,2],[3,4]])
# data2 = torch.tensor([[5,6],[7,8]])
#
# # 方式一
# data = torch.mul(data1, data2)
# print(data)
#
# # 方式二
# data = data1 * data2
# print(data)

data1 = torch.tensor([[1, 2],[3, 4], [5, 6]])
data2 = torch.tensor([[5, 6],[7, 8]])

# 方式一
data = data1 @ data2
print(data)

# 方式二
data = torch.matmul(data1, data2)
print(data)