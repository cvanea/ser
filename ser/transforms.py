from torchvision import transforms

def get_transforms(mean = (0.5,), std = (0.5,)):
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )