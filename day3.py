import torch

#信息查看
# print(torch.__version__)
# print(torch.cuda.is_available())
#print(help(torch.cuda.is_available))

#张量练习
z=torch.zeros(2,5,dtype=torch.int32)
print(z)
print(z.dtype)#创建指定张量

torch.manual_seed(23)
p=torch.rand(3,3)
print(p)
print(p.dtype)#创建随机张量

