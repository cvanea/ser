from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ser.model import Net
from ser.transforms import get_transforms
from ser.train import train as training_loop
import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(2, help="Number of epochs to train."),
    batch_size: int = typer.Option(1000, help="Batch size."),
    learning_rate: float = typer.Option(0.01, help="Learning rate."),
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!
    with open(PROJECT_ROOT / "experiments" / name / "params.txt", "w") as f:
        f.write(f"name: {name}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"learning_rate: {learning_rate}\n")

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = get_transforms()

    # dataloaders
    training_dataloader = training_dataloader(batch_size, ts)
    validation_dataloader = validation_dataloader(batch_size, ts)

    training_loop()


@main.command()
def infer():

    print("This is where the inference code will go")
