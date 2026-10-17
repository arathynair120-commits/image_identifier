
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# Load original test scores
old = np.load("outputs/ood_scores.npz")
known_old = old["known_scores"]
unknown_old = old["unknown_scores"]

# Load recalibrated test scores
new = np.load("outputs/improved_ood_scores.npz")
known_new = new["known_scores"]
unknown_new = new["unknown_scores"]
threshold = float(new["threshold"])

# Graph 1: confidence distributions
plt.figure(figsize=(8, 5))
plt.hist(known_new, bins=30, alpha=0.6, label="Known")
plt.hist(unknown_new, bins=30, alpha=0.6, label="Unknown")
plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.2f}")
plt.xlabel("Maximum Softmax Probability")
plt.ylabel("Number of images")
plt.title("Known vs Unknown Confidence")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/final_confidence.png")
plt.close()

# Graph 2: threshold comparison
labels = ["Original", "Recalibrated"]
known_accept = [
    np.mean(known_old >= 0.87) * 100,
    np.mean(known_new >= threshold) * 100
]
unknown_reject = [
    np.mean(unknown_old < 0.87) * 100,
    np.mean(unknown_new < threshold) * 100
]

x = np.arange(2)
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, known_accept, width, label="Known accepted")
plt.bar(x + width / 2, unknown_reject, width, label="Unknown rejected")
plt.xticks(x, labels)
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)
plt.title("Original vs Recalibrated OOD Detection")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/final_threshold_comparison.png")
plt.close()

print("Saved:")
print("outputs/final_confidence.png")
print("outputs/final_threshold_comparison.png")
