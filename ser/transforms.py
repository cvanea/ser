from torchvision import transforms

class MyTransforms:
    def __init__(self):
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
            ]
        )