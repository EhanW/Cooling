from torch.utils.data import DataLoader
from .datasets import IndexedCIFAR10Train, IndexedCIFAR10Test


def get_train_loader(batch_size, num_workers=4):
	train_full = IndexedCIFAR10Train()
	train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	return train_loader


def get_test_loader(batch_size, num_workers=4):
	test_full = IndexedCIFAR10Test()
	test_loader = DataLoader(test_full, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
	return test_loader


