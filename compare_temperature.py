
import os
import numpy as np
import torch
import torch.nn.functional as F
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

known_train = [
    i for i, y in enumerate(train_set.targets)
    if y in known_classes
]
np.random.seed(42)
np.random.shuffle(known_train)
split = int(0.9 * len(known_train))

val_known_idx = known_train[split:][:500]

val_unknown_idx = []
for c in unknown_classes:
    idx = [i for i, y in enumerate(train_set.targets) if y == c]
    val_unknown_idx.extend(idx[:100])

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

val_known = loader(train_set, val_known_idx)
val_unknown = loader(train_set, val_unknown_idx)
test_known = loader(test_set, test_known_idx)
test_unknown = loader(test_set, test_unknown_idx)

model = CNN(num_classes=5).to(device)
model.load_state_dict(
    torch.load("models/cnn_improved.pth", map_location=device)
)
model.eval()

def get_scores(data_loader, method, temperature):
    scores = []

    with torch.no_grad():
        for images, _ in data_loader:
            logits = model(images.to(device))
            scaled = logits / temperature

            if method == "MSP":
                score = torch.softmax(scaled, dim=1).max(dim=1).values
            else:
                score = torch.logsumexp(scaled, dim=1)

            scores.extend(score.cpu().numpy())

    return np.array(scores)

def calculate_auc(known_scores, unknown_scores):
    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([
        np.ones(len(known_scores)),
        np.zeros(len(unknown_scores))
    ])
    return roc_auc_score(labels, scores)

best = {"auc": -1}

print("Tuning on validation data...")

for method in ["MSP", "Energy"]:
    for temperature in [1, 2, 5, 10, 20, 50, 100]:
        k = get_scores(val_known, method, temperature)
        u = get_scores(val_unknown, method, temperature)

        # Higher score must indicate a known image.
        if method == "Energy":
            k, u = -k, -u

        auc = calculate_auc(k, u)
        print(f"{method}, T={temperature}, Validation AUROC={auc:.4f}")

        if auc > best["auc"]:
            best = {
                "method": method,
                "temperature": temperature,
                "auc": auc
            }

print("\nBest validation configuration:", best)

print("\nEvaluating on test data...")
k = get_scores(test_known, best["method"], best["temperature"])
u = get_scores(test_unknown, best["method"], best["temperature"])

if best["method"] == "Energy":
    k, u = -k, -u

test_auc = calculate_auc(k, u)

print(f"Selected method: {best['method']}")
print(f"Selected temperature: {best['temperature']}")
print(f"Validation AUROC: {best['auc']:.4f}")
print(f"Test AUROC: {test_auc:.4f}")
print("Previous best ODIN test AUROC: 0.6525")

os.makedirs("outputs", exist_ok=True)
np.savez(
    "outputs/temperature_results.npz",
    method=best["method"],
    temperature=best["temperature"],
    validation_auc=best["auc"],
    test_auc=test_auc
)
print("Results saved to outputs/temperature_results.npz")
