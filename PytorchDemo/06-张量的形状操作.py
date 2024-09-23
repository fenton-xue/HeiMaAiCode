import torch
import numpy as np

# data = torch.tensor([[10, 20, 30], [40, 50, 60]])
#
# # 1、使用shape属性或者size方法可以获取tensor的形状
# print(data)
# print(data.shape)
# print(data.size())
#
# # 2、使用reshape 修改张量形状
# new_data = data.reshape(1, 6)
# print(new_data)
# print(new_data.shape)

# mydata1 = torch.tensor([1, 2, 3, 4, 5])
# print("mydata1: ", mydata1.shape, "\n", mydata1)
#
# mydata2 = mydata1.unsqueeze(dim=0)
# print("在0维度上，拓展维度：", mydata2.shape, "\n", mydata2)
#
# mydata3 = mydata1.unsqueeze(dim=1)
# print("在1维度上，拓展维度：", mydata3.shape, "\n", mydata3)
#
# mydata4 = mydata1.unsqueeze(dim=-1)
# print("在最后一个维度上，拓展维度：", mydata4.shape, "\n", mydata4)
#
# mydata5 = mydata4.squeeze()
# print("压缩维度：", mydata5.shape, "\n", mydata5)

# data = torch.tensor(np.random.randint(0, 10, [3, 4, 5]))
# print("data shape: ", data.shape)
#
# # 1、交换第2个和第3个维度
# mydata2 = torch.transpose(data, 1, 2)
# print("mydata2 shape: ", mydata2.shape)
#
# # 2、将data的形状修改为(4, 5, 3)，需要变换多次
# mydata3 = torch.transpose(data, 0, 1)
# mydata4 = torch.transpose(mydata3, 1, 2)
# print("mydata4 shape: ", mydata4.shape)
#
# # 3、使用 permute 将形状修改为 (4, 5, 3)
# # 方式一
# mydata5 = torch.permute(data, [1, 2, 0])
# print("mydata5 shape: ", mydata5.shape)
# # 方式二
# mydata6 = data.permute([1, 2, 0])
# print("mydata6 shape: ", mydata6.shape)

# 若要使用view函数，需要使用contiguous() 变成连续后再使用view
data = torch.tensor([[10, 20, 30], [40, 50, 60]])
print(data.shape)
# 1、判断张量是否使用整块内存，如果不是则调用contiguous变成连续
if data.is_contiguous() is not True:
    data.contiguous()

# 2、使用view()
mydata2 = data.view(3, 2)
print(mydata2.shape)



