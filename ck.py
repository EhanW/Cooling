import torch
from data import IndexedCIFAR10Train
#train_set = IndexedCIFAR10Train()
#a = train_set.cifar10train[50000]

a = torch.rand(10) > 0.5
b = torch.tensor([2, 4, 6])
print(a)
a[b] = True
print(a)
