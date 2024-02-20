from pathlib import Path
import torch
import torch.nn.functional as F
import typer
import os
from datetime import datetime
import json 
from dataclasses import dataclass, asdict
from git import Repo

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

repo = Repo(PROJECT_ROOT)

@dataclass
class Hyperparams:
    epochs: int 
    batch_size: int 
    learning_rate: float 
    commit_hash: str

    def save(self, path):
        '''
        Save hyperparams to a JSON file using the specified save path.
        '''
        with open(path, 'w') as f:
                json.dump(asdict(self), f)


class MyTraining:

    def __init__(self, epochs, device, model, optimizer, training_dataloader, validation_dataloader, batch_size, learning_rate, name):
        self.epochs = epochs
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.name = name
    

    def train(self):

        train_datetime = datetime.now()
  
        # Define save paths
        day = train_datetime.strftime("%d")
        month = train_datetime.strftime("%m")
        year = train_datetime.strftime("%Y")
        hour = train_datetime.strftime("%H")
        minute = train_datetime.strftime("%M")
        separator = "_"
        fname = separator.join([day, month, year, hour, minute])
        run_dir = os.path.join(PROJECT_ROOT, "runs", self.name)

        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)

        print(run_dir)
        print(fname)

        # Do training
        for epoch in range(self.epochs):

            max_val_acc = 0

            for i, (images, labels) in enumerate(self.training_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = F.nll_loss(output, labels)
                loss.backward()
                self.optimizer.step()
                print(
                    f"Train Epoch: {epoch} | Batch: {i}/{len(self.training_dataloader)} "
                    f"| Loss: {loss.item():.4f}"
                )
                # validate
                val_loss = 0
                correct = 0
                with torch.no_grad():
                    for images, labels in self.validation_dataloader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        self.model.eval()
                        output = self.model(images)
                        val_loss += F.nll_loss(output, labels, reduction="sum").item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(labels.view_as(pred)).sum().item()
                    val_loss /= len(self.validation_dataloader.dataset)
                    val_acc = correct / len(self.validation_dataloader.dataset)

                    print(
                        f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
                    )

                    if val_acc > max_val_acc:
                        print(f"New max val accuracy recorded: save weights.")
                        max_val_acc = val_acc
                        # Save model weights 
                        torch.save(self.model, os.path.join(run_dir, fname+'.pth'))
                        
            
            # Save hyperparameters at the end of training
            # Get latest commit associated with train.py and add to hyperparams
            latest_commit = str(list(repo.iter_commits(all=True, max_count=1, paths='ser/train.py'))[0])
            print(latest_commit)

            hp = Hyperparams(self.epochs, self.batch_size, self.learning_rate, latest_commit)
            hp.save(os.path.join(run_dir, fname+'.json'))
            