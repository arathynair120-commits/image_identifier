
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from models.cnn import CNN

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

known = [0, 1, 2, 3, 4]
unknown = [5, 6, 7, 8, 9]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(
    root="data", train=True, download=False, transform=transform
)
test_set = torchvision.datasets.CIFAR10(
    root="data", train=False, download=False, transform=transform
)

model = CNN(num_classes=5).to(device)
model.load_state_dict(torch.load("models/cnn.pth", map_location=device))
model.eval()

def get_scores(dataset):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    scores, predictions, labels = [], [], []

    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(dim=1)

            scores.extend(confidence.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            labels.extend(targets.numpy())

    return np.array(scores), np.array(predictions), np.array(labels)

# Calibration data: known validation images and unseen-in-training classes
known_indices = [
    i for i, (_, label) in enumerate(train_set)
    if label in known
]
generator = torch.Generator().manual_seed(42)
permutation = torch.randperm(len(known_indices), generator=generator).tolist()
val_indices = [known_indices[i] for i in permutation[int(0.9 * len(permutation)):]]

unknown_indices = [
    i for i, (_, label) in enumerate(train_set)
    if label in unknown
]
unknown_indices = unknown_indices[:5000]

known_val = Subset(train_set, val_indices)
unknown_val = Subset(train_set, unknown_indices)

known_scores, _, _ = get_scores(known_val)
unknown_scores, _, _ = get_scores(unknown_val)

# Select threshold using calibration data, not the test set
best_threshold = 0.5
best_balanced_accuracy = -1

for threshold in np.arange(0.1, 1.0, 0.01):
    known_accept = np.mean(known_scores >= threshold)
    unknown_reject = np.mean(unknown_scores < threshold)
    balanced_accuracy = (known_accept + unknown_reject) / 2

    if balanced_accuracy > best_balanced_accuracy:
        best_balanced_accuracy = balanced_accuracy
        best_threshold = threshold

# Final evaluation on the official test split
known_test_indices = [
    i for i, (_, label) in enumerate(test_set) if label in known
]
unknown_test_indices = [
    i for i, (_, label) in enumerate(test_set) if label in unknown
]

known_scores, known_predictions, known_labels = get_scores(
    Subset(test_set, known_test_indices)
)
unknown_scores, _, _ = get_scores(
    Subset(test_set, unknown_test_indices)
)

# CIFAR-10 labels 0-4 match the model's output indices
known_accuracy = np.mean(known_predictions == known_labels)
known_acceptance = np.mean(known_scores >= best_threshold)
accepted = known_scores >= best_threshold

accepted_accuracy = (
    np.mean(known_predictions[accepted] == known_labels[accepted])
    if accepted.any() else 0.0
)

unknown_recall = np.mean(unknown_scores < best_threshold)
unknown_false_acceptance = np.mean(unknown_scores >= best_threshold)

all_scores = np.concatenate([known_scores, unknown_scores])
ood_labels = np.concatenate([
    np.zeros(len(known_scores)),
    np.ones(len(unknown_scores))
])
ood_auc = roc_auc_score(ood_labels, -all_scores)

print("\n--- OOD Evaluation Results ---")
print(f"Chosen confidence threshold: {best_threshold:.2f}")
print(f"Known-class accuracy without rejection: {known_accuracy:.2%}")
print(f"Known images accepted: {known_acceptance:.2%}")
print(f"Accuracy among accepted known images: {accepted_accuracy:.2%}")
print(f"Unknown images correctly rejected: {unknown_recall:.2%}")
print(f"Unknown images incorrectly accepted: {unknown_false_acceptance:.2%}")
print(f"OOD AUROC: {ood_auc:.4f}")
print(f"Calibration balanced accuracy: {best_balanced_accuracy:.2%}")
import os
os.makedirs("outputs", exist_ok=True)

np.savez(
    "outputs/ood_scores.npz",
    known_scores=known_scores,
    unknown_scores=unknown_scores
)
print("Scores saved to outputs/ood_scores.npz")