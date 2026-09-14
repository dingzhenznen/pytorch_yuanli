import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18,efficientnet_b0 


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODEL_PATH = SCRIPT_DIR / "resnet18_cifar10.pth"
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform)



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def build_model(num_classes: int = 10, freeze_backbone: bool = False) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train(epochs: int, batch_size: int, learning_rate: float, freeze_backbone: bool) -> None:
    device = get_device()
    train_loader, test_loader = build_dataloaders(batch_size)

    # dataiter = iter(test_loader)
    # images, labels = next(dataiter)
    # print(images.shape)

    model = build_model(num_classes=len(CIFAR10_CLASSES), freeze_backbone=freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs} - loss: {train_loss:.4f} - test accuracy: {accuracy:.2%}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune torchvision ResNet18 on CIFAR10.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Only train the final classifier layer.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.epochs, args.batch_size, args.lr, args.freeze_backbone)
