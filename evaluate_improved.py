
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from models.cnn import CNN

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(
    root="data", train=True, download=False, transform=transform
)
test_set = torchvision.datasets.CIFAR10(
    root="data", train=False, download=False, transform=transform
)

known_classes = [0, 1, 2, 3, 4]
unknown_classes = [5, 6, 7, 8, 9]

# Recreate the validation split used during training
indices = [
    i for i, label in enumerate(train_set.targets)
    if label in known_classes
]
np.random.shuffle(indices)
split = int(0.9 * len(indices))
cal_known = indices[split:]

# Balance unknown calibration samples across all five classes
cal_unknown = []
for cls in unknown_classes:
    cls_indices = [
        i for i, label in enumerate(train_set.targets)
        if label == cls
    ]
    cal_unknown.extend(
        np.random.choice(cls_indices, 500, replace=False).tolist()
    )

test_known = [
    i for i, label in enumerate(test_set.targets)
    if label in known_classes
]
test_unknown = [
    i for i, label in enumerate(test_set.targets)
    if label in unknown_classes
]

model = CNN(num_classes=5).to(device)
model.load_state_dict(
    torch.load("models/cnn_improved.pth", map_location=device)
)
model.eval()

def get_outputs(dataset, selected_indices):
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=128, shuffle=False
    )
    scores = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            confidence, predictions = probs.max(dim=1)

            scores.extend(confidence.numpy())

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return np.array(scores), correct / total if total else 0

known_cal, _ = get_outputs(train_set, cal_known)
unknown_cal, _ = get_outputs(train_set, cal_unknown)

thresholds = np.arange(0.10, 1.00, 0.01)
best_threshold = 0.5
best_balanced = 0

cal_labels = np.r_[
    np.ones(len(known_cal)),
    np.zeros(len(unknown_cal))
]
cal_scores = np.r_[known_cal, unknown_cal]

for threshold in thresholds:
    predicted_known = (cal_scores >= threshold).astype(int)
    score = balanced_accuracy_score(cal_labels, predicted_known)

    if score > best_balanced:
        best_balanced = score
        best_threshold = threshold

known_scores, known_accuracy = get_outputs(test_set, test_known)
unknown_scores, _ = get_outputs(test_set, test_unknown)

accepted = known_scores >= best_threshold
accepted_accuracy = (
    np.mean(accepted) if len(accepted) else 0
)
accepted_correct = (
    np.mean(accepted)  # acceptance rate
)

# Compute actual accuracy among accepted known images
# Re-run known test set to collect correct predictions
known_loader = DataLoader(
    Subset(test_set, test_known), batch_size=128, shuffle=False
)
correct_accepted = 0
total_accepted = 0

with torch.no_grad():
    offset = 0
    for images, labels in known_loader:
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        confidence, predictions = probs.max(dim=1)
        mask = confidence >= best_threshold

        correct_accepted += (
            (predictions[mask] == labels[mask]).sum().item()
        )
        total_accepted += mask.sum().item()

accepted_accuracy = (
    correct_accepted / total_accepted if total_accepted else 0
)

unknown_rejection = np.mean(unknown_scores < best_threshold)
false_acceptance = np.mean(unknown_scores >= best_threshold)

labels = np.r_[
    np.ones(len(known_scores)),
    np.zeros(len(unknown_scores))
]
all_scores = np.r_[known_scores, unknown_scores]
auc = roc_auc_score(labels, all_scores)

print("\n--- Improved Model Evaluation ---")
print(f"Known test accuracy: {known_accuracy * 100:.2f}%")
print(f"Selected threshold: {best_threshold:.2f}")
print(f"Calibration balanced accuracy: {best_balanced * 100:.2f}%")
print(f"Known images accepted: {np.mean(accepted) * 100:.2f}%")
print(f"Accepted-known accuracy: {accepted_accuracy * 100:.2f}%")
print(f"Unknown images rejected: {unknown_rejection * 100:.2f}%")
print(f"Unknown false acceptance: {false_acceptance * 100:.2f}%")
print(f"OOD AUROC: {auc:.4f}")
