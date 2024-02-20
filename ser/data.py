    # dataloaders
from torch.utils.data import DataLoader
import datasets


def training_dataloader(batch_size, ts):
    
    return DataLoader(
        datasets.MNIST(root="../data", download=True, train=True, transform=ts),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

def validation_dataloader(batch_size, ts): 
    return DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )