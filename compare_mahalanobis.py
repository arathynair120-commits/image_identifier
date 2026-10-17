
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from models.cnn import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

known_classes = [0, 1, 2, 3, 4]
unknown_classes = [5, 6, 7, 8, 9]

train_idx = [
    i for i, y in enumerate(train_set.targets)
    if y in known_classes
]

test_known_idx = [
    i for i, y in enumerate(test_set.targets)
    if y in known_classes
]

test_unknown_idx = [
    i for i, y in enumerate(test_set.targets)
    if y in unknown_classes
]

def loader(dataset, indices):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=128,
        shuffle=False
    )

train_loader = loader(train_set, train_idx)
known_loader = loader(test_set, test_known_idx)
unknown_loader = loader(test_set, test_unknown_idx)

model = CNN(num_classes=5).to(device)
model.load_state_dict(
    torch.load("models/cnn_improved.pth", map_location=device)
)
model.eval()

def extract_features(data_loader):
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            x = model.features(images)
            x = model.classifier[0](x)
            x = model.classifier[1](x)
            x = model.classifier[2](x)

            features.append(x.cpu())
            labels.append(targets)

    return torch.cat(features), torch.cat(labels)

print("Extracting training features...")
train_features, train_labels = extract_features(train_loader)

means = []
for c in known_classes:
    means.append(train_features[train_labels == c].mean(dim=0))

means = torch.stack(means)

print("Calculating covariance...")
residuals = train_features - means[train_labels]
covariance = residuals.T @ residuals / (len(residuals) - len(known_classes))

ridge = 0.01 * covariance.diag().mean()
covariance += ridge * torch.eye(covariance.shape[0])

precision = torch.linalg.pinv(covariance).to(device)
means = means.to(device)

def mahalanobis_scores(data_loader):
    scores = []

    features, _ = extract_features(data_loader)
    features = features.to(device)

    for batch in features.split(512):
        distances = []

        for mean in means:
            diff = batch - mean
            distance = (diff @ precision * diff).sum(dim=1)
            distances.append(distance)

        min_distance = torch.stack(distances, dim=1).min(dim=1).values
        scores.extend((-min_distance).cpu().numpy())

    return np.array(scores)

print("Scoring known test images...")
known_scores = mahalanobis_scores(known_loader)

print("Scoring unknown test images...")
unknown_scores = mahalanobis_scores(unknown_loader)

scores = np.concatenate([known_scores, unknown_scores])
labels = np.concatenate([
    np.ones(len(known_scores)),
    np.zeros(len(unknown_scores))
])

auc = roc_auc_score(labels, scores)

print("\n--- Mahalanobis OOD Results ---")
print(f"Mahalanobis Test AUROC: {auc:.4f}")
print("Previous MSP AUROC: 0.6261")
print("Previous MaxLogit AUROC: 0.6431")
print("Previous Energy AUROC: 0.6435")
print("Previous ODIN AUROC: 0.6525")

os.makedirs("outputs", exist_ok=True)

np.savez(
    "outputs/mahalanobis_results.npz",
    known_scores=known_scores,
    unknown_scores=unknown_scores,
    test_auc=auc
)

print("Results saved to outputs/mahalanobis_results.npz")
