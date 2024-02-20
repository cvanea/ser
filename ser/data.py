from torch.utils.data import DataLoader
from torchvision import datasets

class MyDataLoader():
    def __init__(self, data_dir, batch_size, transforms):
        self.training_dataloader = DataLoader(
            datasets.MNIST(root="../data", download=True, train=True, transform=transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )

        self.validation_dataloader = DataLoader(
            datasets.MNIST(root=data_dir, download=True, train=False, transform=transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
        )