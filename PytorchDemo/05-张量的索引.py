import torch

data = torch.randint(0, 10, [4, 5])
print(data)

# print(data[0])
# print(data[:, 0])

# 返回 （0，1）（1，2）两个位置的元素
# print(data[[0, 1], [1, 2]])
#
# # 返回 0、1行的 1、2列共4个元素
# print(data[[[0], [1]], [1, 2]])

# # 前3行的前2列数据
# print(data[:3, :2])
#
# # 第2行到最后的前2列数据
# print(data[2:, :2])


# data = torch.randint(0, 10, [3, 4, 5, 6])
# print(data)
#
# # 获取 0 轴上的第一个数据
# print(data[0, :, :, :])
#
# # 获取 1 轴上的第一个数据
# print(data[:, 0, :, :])
#
# # 获取 2 轴上的第一个数据
# print(data[:, :, 0, :])
# print(data[:, :, :, 0])