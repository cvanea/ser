from ser.model import Net
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import torch
from torch import optim
from ser.data import training_dataloader, validation_dataloader
import torch.nn.functional as F
from ser.transforms import get_transforms


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
    # epochs = epochs
    # batch_size = batch_size
    # learning_rate = learning_rate

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

    # train
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )
            # validate
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for images, labels in validation_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    model.eval()
                    output = model(images)
                    val_loss += F.nll_loss(output, labels, reduction="sum").item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                val_loss /= len(validation_dataloader.dataset)
                val_acc = correct / len(validation_dataloader.dataset)

                print(
                    f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
                )
