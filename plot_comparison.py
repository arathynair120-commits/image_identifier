
import numpy as np
import matplotlib.pyplot as plt

data = np.load("outputs/ood_comparison.npz")

known_msp = data["known_msp"]
unknown_msp = data["unknown_msp"]
known_energy = data["known_energy"]
unknown_energy = data["unknown_energy"]

plt.figure(figsize=(8, 5))
plt.hist(known_msp, bins=30, alpha=0.6, label="Known images")
plt.hist(unknown_msp, bins=30, alpha=0.6, label="Unknown images")
plt.xlabel("Maximum Softmax Probability")
plt.ylabel("Number of images")
plt.title("MSP Confidence Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/msp_distribution.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(known_energy, bins=30, alpha=0.6, label="Known images")
plt.hist(unknown_energy, bins=30, alpha=0.6, label="Unknown images")
plt.xlabel("Energy Score (log-sum-exp)")
plt.ylabel("Number of images")
plt.title("Energy Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/energy_distribution.png")
plt.show()

print("Graphs saved in outputs/")
