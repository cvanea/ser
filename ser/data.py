from torch.utils.data import DataLoader
from ser.transforms import transform_data
from torchvision import datasets
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def get_train_dataloader(batch_size):
    # dataloaders
    training_dataloader = DataLoader(
        datasets.MNIST(root="../data", download=True, train=True, transform=transform_data()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    return training_dataloader

def get_validation_dataloader(batch_size):
    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=transform_data()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    return validation_dataloader
     