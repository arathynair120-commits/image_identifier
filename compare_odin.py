
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

# Recreate the known-class validation split
known_train = [
    i for i, label in enumerate(train_set.targets)
    if label in known_classes
]

np.random.seed(42)
np.random.shuffle(known_train)

split = int(0.9 * len(known_train))
val_known = known_train[split:][:500]

# Balanced unknown validation samples
val_unknown = []
for c in unknown_classes:
    indices = [
        i for i, label in enumerate(train_set.targets)
        if label == c
    ]
    val_unknown.extend(indices[:100])

# Official test indices, used only after tuning
test_known = [
    i for i, label in enumerate(test_set.targets)
    if label in known_classes
]
test_unknown = [
    i for i, label in enumerate(test_set.targets)
    if label in unknown_classes
]

def loader(dataset, indices):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=128,
        shuffle=False
    )

val_known_loader = loader(train_set, val_known)
val_unknown_loader = loader(train_set, val_unknown)
test_known_loader = loader(test_set, test_known)
test_unknown_loader = loader(test_set, test_unknown)

model = CNN(num_classes=5).to(device)
model.load_state_dict(
    torch.load("models/cnn_improved.pth", map_location=device)
)
model.eval()


def odin_scores(data_loader, temperature, epsilon):
    scores = []

    for images, _ in data_loader:
        images = images.to(device).detach()
        images.requires_grad_(True)

        model.zero_grad(set_to_none=True)

        logits = model(images)
        scaled_logits = logits / temperature
        predicted = scaled_logits.argmax(dim=1)

        loss = F.cross_entropy(scaled_logits, predicted)
        loss.backward()

        gradient = images.grad.detach().sign()
        adjusted = (images - epsilon * gradient).detach()

        with torch.no_grad():
            logits = model(adjusted)
            probabilities = torch.softmax(
                logits / temperature, dim=1
            )
            scores.extend(
                probabilities.max(dim=1).values.cpu().numpy()
            )

    return np.array(scores)


def auc_score(known_scores, unknown_scores):
    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([
        np.ones(len(known_scores)),
        np.zeros(len(unknown_scores))
    ])
    return roc_auc_score(labels, scores)


print("Tuning ODIN on validation data...")

best_auc = -1
best_temperature = 1
best_epsilon = 0

temperatures = [1, 10, 100, 1000]
epsilons = [0, 0.0005, 0.001, 0.002, 0.004]

for temperature in temperatures:
    for epsilon in epsilons:
        known_scores = odin_scores(
            val_known_loader, temperature, epsilon
        )
        unknown_scores = odin_scores(
            val_unknown_loader, temperature, epsilon
        )

        auc = auc_score(known_scores, unknown_scores)

        print(
            f"T={temperature}, epsilon={epsilon:.4f}, "
            f"Validation AUROC={auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            best_temperature = temperature
            best_epsilon = epsilon

print("\nBest ODIN settings:")
print("Temperature:", best_temperature)
print("Epsilon:", best_epsilon)
print(f"Validation AUROC: {best_auc:.4f}")

print("\nEvaluating selected settings on test data...")

known_scores = odin_scores(
    test_known_loader, best_temperature, best_epsilon
)
unknown_scores = odin_scores(
    test_unknown_loader, best_temperature, best_epsilon
)

odin_auc = auc_score(known_scores, unknown_scores)

print(f"ODIN Test AUROC: {odin_auc:.4f}")
print(f"Previous Energy AUROC: 0.6435")

os.makedirs("outputs", exist_ok=True)

np.savez(
    "outputs/odin_results.npz",
    known_scores=known_scores,
    unknown_scores=unknown_scores,
    temperature=best_temperature,
    epsilon=best_epsilon,
    validation_auc=best_auc,
    test_auc=odin_auc
)

print("Results saved to outputs/odin_results.npz")
