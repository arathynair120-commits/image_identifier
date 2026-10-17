
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from models.cnn import CNN
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

known_classes = [0, 1, 2, 3, 4]
unknown_classes = [5, 6, 7, 8, 9]

train_set = torchvision.datasets.CIFAR10(
    root="data", train=True, download=False, transform=transform
)

test_set = torchvision.datasets.CIFAR10(
    root="data", train=False, download=False, transform=transform
)

def balanced_indices(dataset, classes, samples_per_class):
    labels = np.array(dataset.targets)
    selected = []

    for cls in classes:
        indices = np.where(labels == cls)[0]
        chosen = np.random.choice(
            indices, samples_per_class, replace=False
        )
        selected.extend(chosen)

    return selected

# Calibration samples from the training dataset
cal_known = balanced_indices(train_set, known_classes, 500)
cal_unknown = balanced_indices(train_set, unknown_classes, 500)

# Test samples remain separate from calibration
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
    torch.load("models/cnn.pth", map_location=device)
)
model.eval()

def get_scores(dataset, indices):
    loader = DataLoader(
        Subset(dataset, indices), batch_size=128, shuffle=False
    )
    scores = []

    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            confidence = torch.softmax(logits, dim=1).max(dim=1).values
            scores.extend(confidence.cpu().numpy())

    return np.array(scores)

known_cal_scores = get_scores(train_set, cal_known)
unknown_cal_scores = get_scores(train_set, cal_unknown)

thresholds = np.arange(0.10, 1.00, 0.01)
best_threshold = 0.5
best_score = -1

for threshold in thresholds:
    predictions = np.r_[
        known_cal_scores >= threshold,
        unknown_cal_scores < threshold
    ]
    actual = np.r_[
        np.ones(len(known_cal_scores)),
        np.ones(len(unknown_cal_scores))
    ]

    # Labels: 1 = correctly accepted known or rejected unknown
    actual = np.r_[
        np.ones(len(known_cal_scores)),
        np.zeros(len(unknown_cal_scores))
    ]
    predicted = np.r_[
        known_cal_scores >= threshold,
        unknown_cal_scores >= threshold
    ]

    # Binary label: 1 = known, 0 = unknown
    actual = np.r_[
        np.ones(len(known_cal_scores)),
        np.zeros(len(unknown_cal_scores))
    ]
    score = balanced_accuracy_score(actual, predicted)

    if score > best_score:
        best_score = score
        best_threshold = threshold

known_test_scores = get_scores(test_set, test_known)
unknown_test_scores = get_scores(test_set, test_unknown)

known_accept = np.mean(known_test_scores >= best_threshold)
unknown_reject = np.mean(unknown_test_scores < best_threshold)
accepted_known_scores = known_test_scores[
    known_test_scores >= best_threshold
]

known_test_labels = np.ones(len(known_test_scores))
unknown_test_labels = np.zeros(len(unknown_test_scores))
all_scores = np.r_[known_test_scores, unknown_test_scores]
all_labels = np.r_[known_test_labels, unknown_test_labels]
auc = roc_auc_score(all_labels, all_scores)

print("\n--- Improved OOD Calibration ---")
print(f"Threshold: {best_threshold:.2f}")
print(f"Calibration balanced accuracy: {best_score:.4f}")
print(f"Known images accepted: {known_accept * 100:.2f}%")
print(f"Unknown images rejected: {unknown_reject * 100:.2f}%")
print(f"MSP AUROC: {auc:.4f}")

os.makedirs("outputs", exist_ok=True)
np.savez(
    "outputs/improved_ood_scores.npz",
    known_scores=known_test_scores,
    unknown_scores=unknown_test_scores,
    threshold=best_threshold
)
print("Scores saved to outputs/improved_ood_scores.npz")
