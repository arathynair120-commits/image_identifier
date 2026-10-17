
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from models.cnn import CNN

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.CIFAR10(
    root="data", train=True, download=True, transform=transform
)

# Known classes: airplane, automobile, bird, cat, deer
known_indices = [
    i for i, (_, label) in enumerate(dataset)
    if label in [0, 1, 2, 3, 4]
]

known_data = Subset(dataset, known_indices)

train_size = int(0.9 * len(known_data))
val_size = len(known_data) - train_size

train_data, val_data = random_split(
    known_data,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128)

model = CNN(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

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
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(
        f"Epoch {epoch + 1}/5 | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Validation accuracy: {100 * correct / total:.2f}%"
    )

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cnn.pth")
print("Model saved to models/cnn.pth")
