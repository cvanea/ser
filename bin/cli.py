import torch
import typer

from bin.model import Net
from bin.data import data_loader
from bin.train import training

main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        ..., "-e", "--epochs", help="Number of epochs."
    ),
    batch_size: int = typer.Option(
        ..., "-b", "--batch_size", help="Batch size."
    ),
    learning_rate: float = typer.Option(
        ..., "-lr", "--learning_rate", help="Learning rate."
    ),
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training(device, name, epochs, batch_size, learning_rate)
    

@main.command()
def infer():
    print("This is where the inference code will go")