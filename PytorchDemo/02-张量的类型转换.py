import torch
import numpy as np

# data = torch.tensor([2, 3, 4])
# data_numpy = data.numpy()
# print(data)
# print(type(data))
# print(data_numpy)
# print(type(data_numpy))

# data_numpy = np.array([2, 3, 4])
# data_tensor = torch.from_numpy(data_numpy)
# print(data_numpy)
# print(type(data_numpy))
# print(data_tensor)
# print(type(data_tensor))
#
#
# data_numpy = np.array([2, 3, 4])
# data_tensor = torch.tensor(data_numpy)
# print(data_numpy)
# print(type(data_numpy))
# print(data_tensor)
# print(type(data_tensor))

data = torch.tensor([30,])
print(data.item())
data = torch.tensor(30)
print(data.item())