
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

test_set = torchvision.datasets.CIFAR10(
    root="data", train=False, download=False, transform=transform
)

known_classes = [0, 1, 2, 3, 4]
unknown_classes = [5, 6, 7, 8, 9]

known_indices = [
    i for i, (_, label) in enumerate(test_set)
    if label in known_classes
]
unknown_indices = [
    i for i, (_, label) in enumerate(test_set)
    if label in unknown_classes
]

known_loader = DataLoader(
    Subset(test_set, known_indices), batch_size=128
)
unknown_loader = DataLoader(
    Subset(test_set, unknown_indices), batch_size=128
)

model = CNN(num_classes=5).to(device)
model.load_state_dict(
    torch.load("models/cnn.pth", map_location=device)
)
model.eval()

def collect_scores(loader):
    msp_scores = []
    energy_scores = []

    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)

            msp = probabilities.max(dim=1).values
            energy = torch.logsumexp(logits, dim=1)

            msp_scores.extend(msp.cpu().numpy())
            energy_scores.extend(energy.cpu().numpy())

    return np.array(msp_scores), np.array(energy_scores)

known_msp, known_energy = collect_scores(known_loader)
unknown_msp, unknown_energy = collect_scores(unknown_loader)

# Higher MSP means more likely known.
msp_auc = roc_auc_score(
    np.r_[np.zeros(len(known_msp)), np.ones(len(unknown_msp))],
    -np.r_[known_msp, unknown_msp]
)

# Higher energy usually indicates more likely known.
energy_auc = roc_auc_score(
    np.r_[np.zeros(len(known_energy)), np.ones(len(unknown_energy))],
    -np.r_[known_energy, unknown_energy]
)

print("\n--- OOD Method Comparison ---")
print(f"MSP AUROC: {msp_auc:.4f}")
print(f"Energy AUROC: {energy_auc:.4f}")

if energy_auc > msp_auc:
    print("Energy scoring performed better on this test set.")
else:
    print("MSP scoring performed as well as or better than energy scoring.")

np.savez(
    "outputs/ood_comparison.npz",
    known_msp=known_msp,
    unknown_msp=unknown_msp,
    known_energy=known_energy,
    unknown_energy=unknown_energy
)
print("Scores saved to outputs/ood_comparison.npz")
