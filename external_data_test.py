import torch, numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from model import BrainTumorCNN

# -----------------------------
# Config
# -----------------------------
EXT_TEST_DIR = r"brain_mri_new_dataset/brisc2025/classification_task/test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Data (external dataset: BRISC 2025)
# -----------------------------
# Only resize + normalization (no augmentation for evaluation)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])

dataset = datasets.ImageFolder(EXT_TEST_DIR, transform=transform)
loader  = DataLoader(dataset, batch_size=32, shuffle=False)

# -----------------------------
# Load best model
# -----------------------------
model = BrainTumorCNN().to(DEVICE)
model.load_state_dict(torch.load("best_brain_tumor_cnn.pth", map_location=DEVICE))
model.eval()

# -----------------------------
# Evaluation
# -----------------------------
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x).argmax(1)
        y_true += y.cpu().tolist()
        y_pred += p.cpu().tolist()

# -----------------------------
# Metrics
# -----------------------------
acc = (np.array(y_true) == np.array(y_pred)).mean()*100
print("Classes:", dataset.classes)
print(f"External Accuracy: {acc:.2f}%")

# Classification report
print("\n", classification_report(y_true, y_pred, target_names=dataset.classes))

# Confusion matrix
print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))