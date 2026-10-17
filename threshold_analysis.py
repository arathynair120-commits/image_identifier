
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Replace these with scores saved from evaluate_ood.py
# Best option: save known_scores and unknown_scores to a file.
data = np.load("outputs/ood_scores.npz")
known_scores = data["known_scores"]
unknown_scores = data["unknown_scores"]

thresholds = np.arange(0.1, 1.0, 0.02)

known_accept = [
    np.mean(known_scores >= t) for t in thresholds
]
unknown_reject = [
    np.mean(unknown_scores < t) for t in thresholds
]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, known_accept, label="Known images accepted")
plt.plot(thresholds, unknown_reject, label="Unknown images rejected")
plt.xlabel("Confidence threshold")
plt.ylabel("Proportion")
plt.title("Effect of Confidence Threshold on OOD Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/threshold_analysis.png")
plt.show()

print("Graph saved to outputs/threshold_analysis.png")

