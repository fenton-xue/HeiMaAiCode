import torch

data1 = torch.randint(0, 10, [1, 2, 3])
data2 = torch.randint(0, 10, [1, 2, 3])
print(data1)
print(data2)

# 1、按0维度拼接
new_data = torch.cat([data1, data2], dim=0)
print(new_data)
print(new_data.shape)

# 2、按1维度拼接
new_data = torch.cat([data1, data2], dim=1)
print(new_data)
print(new_data.shape)

# 2、按1维度拼接
new_data = torch.cat([data1, data2], dim=2)
print(new_data)
print(new_data.shape)