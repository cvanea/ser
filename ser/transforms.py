from torchvision import transforms

def transform_data():
    # torch transforms
    ts = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    return ts