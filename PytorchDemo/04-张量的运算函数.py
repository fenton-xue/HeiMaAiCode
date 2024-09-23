import torch

data = torch.randint(0, 10, [2, 3], dtype=torch.float64)
print(data)

# 1、计算均值
# 注意：tensor必须为float或者double
print(data.mean())

# 2、计算总和
print(data.sum())

# 3、计算平方
print(torch.pow(data, 2))

# 4、计算平方根
print(data.sqrt())

# 5、指数计算，e^n次方
print(data.exp())

# 6、对数计算
print(data.log()) # 以e为底
print(data.log2())
print(data.log10())