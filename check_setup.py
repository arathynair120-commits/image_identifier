import torch
import torchvision
import sklearn
import pandas
import matplotlib

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)
print("GPU available:", torch.cuda.is_available())
print("Setup successful!")
