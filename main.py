# main.py
import torch
from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, dataloader, criterion, optimizer):
    model.train()
    for x, y in dataloader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_ = model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()


def eval_single_epoch(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            preds = torch.argmax(y_, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def train_model(config):

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load training folder with subfolders car/flower
    dataset = MyDataset(config["train_dir"], transform)

    # Split train/val
    total = len(dataset)
    train_len = int(total * config["size_train"])
    val_len   = total - train_len

    train_data, val_data = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=config["batch_size"], shuffle=False)

    model = MyModel(
        n_features=config["features"],
        n_hidden=config["hidden_layers"],
        n_outputs=config["outputs"],
        in_channels=3  # RGB
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        train_single_epoch(model, train_loader, criterion, optimizer)
        acc = eval_single_epoch(model, val_loader)
        print(f"Validation accuracy: {acc:.4f}")

    return model


if __name__ == "__main__":
    config = {
        "epochs": 10,
        "batch_size": 32,
        "train_dir": "/home/anadal/workspace/car_flowers_categorizer/archive/dataset/cars_vs_flowers/training_set/",
        "features": 4096,          # unused but kept for compatibility
        "hidden_layers": 128,
        "outputs": 2,              # car vs flower
        "lr": 0.01,
        "size_train": 0.8,
    }

    train_model(config)
