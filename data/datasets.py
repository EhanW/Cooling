from torchvision import datasets, transforms as T
from torch.utils.data import Dataset


class IndexedCIFAR10Train(Dataset):
    def __init__(self):
        self.cifar10train = datasets.CIFAR10(root='/data/yihan/datasets',
                                             download=False,
                                             train=True,
                                             transform=T.Compose([T.RandomCrop(32, 4),
                                                                  T.RandomHorizontalFlip(),
                                                                  T.ToTensor()]))

    def __getitem__(self, index):
        data, target = self.cifar10train[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10train)


class IndexedCIFAR10Test(Dataset):
    def __init__(self):
        self.cifar10test = datasets.CIFAR10(root='/data/yihan/datasets',
                                            download=False,
                                            train=False,
                                            transform=T.ToTensor())

    def __getitem__(self, index):
        data, target = self.cifar10test[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10test)
