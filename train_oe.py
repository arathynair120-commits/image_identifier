
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from models.cnn import CNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)

known_classes = [0, 1, 2, 3, 4]
unknown_classes = [5, 6, 7, 8, 9]

# Training augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Validation and test transformation
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_aug = torchvision.datasets.CIFAR10(
    root="data", train=True, download=False,
    transform=train_transform
)

train_eval = torchvision.datasets.CIFAR10(
    root="data", train=True, download=False,
    transform=eval_transform
)

test_set = torchvision.datasets.CIFAR10(
    root="data", train=False, download=False,
    transform=eval_transform
)

# Split each class into 90% training and 10% validation
train_known_idx = []
val_known_idx = []
train_ood_idx = []
val_ood_idx = []

for c in known_classes + unknown_classes:
    indices = [
        i for i, label in enumerate(train_aug.targets)
        if label == c
    ]
    np.random.shuffle(indices)
    cut = int(0.9 * len(indices))

    if c in known_classes:
        train_known_idx.extend(indices[:cut])
        val_known_idx.extend(indices[cut:])
    else:
        train_ood_idx.extend(indices[:cut])
        val_ood_idx.extend(indices[cut:])

def make_loader(dataset, indices, shuffle=False):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=128,
        shuffle=shuffle,
        num_workers=0
    )

known_loader = make_loader(train_aug, train_known_idx, True)
ood_loader = make_loader(train_aug, train_ood_idx, True)

val_known_loader = make_loader(train_eval, val_known_idx)
val_ood_loader = make_loader(train_eval, val_ood_idx)

# Load the existing model without changing its checkpoint
model = CNN(num_classes=5).to(device)
model.load_state_dict(
    torch.load("models/cnn_improved.pth", map_location=device)
)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0002, weight_decay=0.0001
)

# Weight assigned to the Outlier Exposure loss
oe_weight = 0.5
epochs = 8
best_auc = -1

def evaluate(loader):
    model.eval()
    scores = []
    labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            scores.extend(probs.max(dim=1).values.cpu().numpy())
            labels.extend(targets.numpy())

            if all(y in known_classes for y in targets.tolist()):
                correct += (
                    logits.argmax(dim=1).cpu() == targets
                ).sum().item()
                total += len(targets)

    return np.array(scores), np.array(labels), correct, total

print("Starting Outlier Exposure training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    batches = 0

    for (known_images, known_labels), (ood_images, _) in zip(
        known_loader, ood_loader
    ):
        known_images = known_images.to(device)
        known_labels = known_labels.to(device)
        ood_images = ood_images.to(device)

        optimizer.zero_grad()

        # Normal classification loss
        known_logits = model(known_images)
        loss_cls = F.cross_entropy(known_logits, known_labels)

        # Encourage uniform predictions for OOD images
        ood_logits = model(ood_images)
        log_probs = F.log_softmax(ood_logits, dim=1)
        uniform = torch.full_like(log_probs, 1.0 / len(known_classes))

        loss_oe = F.kl_div(
            log_probs, uniform, reduction="batchmean"
        )

        loss = loss_cls + oe_weight * loss_oe
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    # Validation: known images should have higher MSP scores
    known_scores, _, correct, total = evaluate(val_known_loader)
    ood_scores, _, _, _ = evaluate(val_ood_loader)

    val_labels = np.concatenate([
        np.ones(len(known_scores)),
        np.zeros(len(ood_scores))
    ])
    val_scores = np.concatenate([known_scores, ood_scores])
    val_auc = roc_auc_score(val_labels, val_scores)

    val_acc = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {total_loss / batches:.4f} | "
        f"Known validation accuracy: {val_acc:.2f}% | "
        f"Validation AUROC: {val_auc:.4f}"
    )

    if val_auc > best_auc:
        best_auc = val_auc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/cnn_oe.pth")

# Evaluate the selected model on the official test set
print("\nEvaluating selected model on test data...")

model.load_state_dict(
    torch.load("models/cnn_oe.pth", map_location=device)
)

test_known_idx = [
    i for i, y in enumerate(test_set.targets)
    if y in known_classes
]
test_ood_idx = [
    i for i, y in enumerate(test_set.targets)
    if y in unknown_classes
]

test_known_loader = make_loader(test_set, test_known_idx)
test_ood_loader = make_loader(test_set, test_ood_idx)

known_scores, known_labels, correct, total = evaluate(test_known_loader)
ood_scores, _, _, _ = evaluate(test_ood_loader)

test_labels = np.concatenate([
    np.ones(len(known_scores)),
    np.zeros(len(ood_scores))
])
test_scores = np.concatenate([known_scores, ood_scores])
test_auc = roc_auc_score(test_labels, test_scores)

print("\n--- Outlier Exposure Results ---")
print(f"Best validation AUROC: {best_auc:.4f}")
print(f"Known test accuracy: {100 * correct / total:.2f}%")
print(f"Test AUROC: {test_auc:.4f}")
print("Previous ODIN AUROC: 0.6525")

os.makedirs("outputs", exist_ok=True)
np.savez(
    "outputs/oe_results.npz",
    known_scores=known_scores,
    unknown_scores=ood_scores,
    validation_auc=best_auc,
    test_auc=test_auc
)

print("Model saved to models/cnn_oe.pth")
print("Results saved to outputs/oe_results.npz")
