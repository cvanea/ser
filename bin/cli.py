import torch
import typer

from ser.data import data_loader
from ser.transforms import transform
from ser.model import Net
from ser.data import data_loader
from ser.train import training
from ser.model import Net

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

    # load model
    model = Net().to(device)
    # transforms 
    ts = transform()
    # dataloaders
    training_dataloader, validation_dataloader = data_loader(batch_size, ts)
    # train 
    training(model, device, name, epochs, batch_size, learning_rate, training_dataloader, validation_dataloader)
    

@main.command()
def infer():
    print("This is where the inference code will go")