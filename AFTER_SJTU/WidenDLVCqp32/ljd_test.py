import torch

a = torch.randn(2,1,4,4)
b = torch.cat((a,a,a),1)
print('a:',a)
print('b:',b)