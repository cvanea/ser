from pathlib import Path
import torch
from torch import optim
import typer
from ser.model import Net
from ser.transforms import MyTransforms
from ser.data import MyDataLoader
from ser.train import MyTraining
import torchvision.models as models

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        2, "-e", "--epochs", help="Number of epochs to train for."
    ),
    batch_size: int = typer.Option(
        1000, "-bs", "--batch_size", help="Batch size."
    ),
    learning_rate: float = typer.Option(
        0.1, "-lr", "--learning_rate", help="Starting learning rate"
    )
):
    print(f"Running {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    #model = Net().to(device)
    model = models.vgg16()

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = MyTransforms().train_transforms

    # dataloaders
    dl = MyDataLoader(DATA_DIR, batch_size, ts)

    # train
    tr = MyTraining(
        epochs, 
        device, 
        model, 
        optimizer, 
        dl.training_dataloader, 
        dl.validation_dataloader,
        batch_size,
        learning_rate,
        name)
    
    tr.train()

@main.command()
def infer():
    print("This is where the inference code will go")
