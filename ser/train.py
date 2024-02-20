from ser.data import *
import torch
from torch import optim
import torch.nn.functional as F

def train_model(name, epochs, learning_rate, device, train_dataloader, validation_dataloader, model):

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # record highest validation accuracy
    highest_accuracy = 0
    best_epoch = 0

    # train
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(train_dataloader)} "
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

                # update highest accuracy and corresponding epoch
                if val_acc > highest_accuracy:
                    highest_accuracy = val_acc
                    best_epoch = epoch

    # save the model
    torch.save(model, f"ser/results/{name}/model.pt")

    # save highest accuracy and best epoch to a text file
    with open(f"ser/results/{name}/results.txt", 'w') as f:
        f.write(f"Highest Accuracy: {highest_accuracy}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
            
    
        