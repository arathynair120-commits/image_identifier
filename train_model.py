
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from models.cnn import CNN

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Augmentation for training images
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# No augmentation for validation images
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(
    root="data", train=True, download=False,
    transform=train_transform
)

val_data = torchvision.datasets.CIFAR10(
    root="data", train=True, download=False,
    transform=val_transform
)

# Keep only the five known classes
known_classes = [0, 1, 2, 3, 4]

indices = [
    i for i, label in enumerate(train_data.targets)
    if label in known_classes
]

np.random.shuffle(indices)
split = int(0.9 * len(indices))

train_indices = indices[:split]
val_indices = indices[split:]

train_loader = DataLoader(
    Subset(train_data, train_indices),
    batch_size=128, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    Subset(val_data, val_indices),
    batch_size=128, shuffle=False, num_workers=0
)

model = CNN(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.001, weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=6, gamma=0.5
)

best_accuracy = 0.0
os.makedirs("models", exist_ok=True)

for epoch in range(15):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    scheduler.step()

    print(
        f"Epoch {epoch + 1}/15 | "
        f"Loss: {avg_loss:.4f} | "
        f"Validation accuracy: {accuracy:.2f}%"
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "models/cnn_improved.pth")

print(f"Best validation accuracy: {best_accuracy:.2f}%")
print("Improved model saved to models/cnn_improved.pth")
