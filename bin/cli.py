from pathlib import Path
from ser.train import training_function

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        default = 2, help="Number of epochs for experiment."
    ),
    batch_size: int = typer.Option(
        default = 1000, help="Batch size for experiment."
    ),
    learning_rate: float = typer.Option(
        default = 0.01, help="Learning rate for experiment."
    ),
):

    
    print(f"Running experiment {name}")
    # train
    training_function(epochs, batch_size, learning_rate)


@main.command()
def infer():
    print("This is where the inference code will go")
