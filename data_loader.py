import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Image size and batch size used consistently across train/val/test
IMG_SIZE = 224
BATCH_SIZE = 32

# Fixed normalization (kept consistent between training and evaluation)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# -----------------------------
# Transforms
# -----------------------------
# Training transforms: light, medically-safe augmentation (no random crop to avoid cutting lesions)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Validation/Test transforms: no augmentation (deterministic evaluation)
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# -----------------------------
# Custom Dataset
# -----------------------------
class BrainMRIDataset(Dataset):
    """
        CSV-driven dataset for brain MRI classification.
        Expects a CSV with columns:
            - path: filesystem path to the image
            - label: class name (e.g., 'glioma', 'meningioma', 'notumor', 'pituitary')
        Label indices are built once from the unique sorted label names to keep mapping stable.
        """
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # Build a deterministic label mapping (alphabetical by class name)
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}

    def __len__(self):
        return len(self.data) # Number of samples

    def __getitem__(self, idx):
        # Fetch image path and label string
        img_path = self.data.iloc[idx]['path']
        label = self.label_map[self.data.iloc[idx]['label']]
        image = Image.open(img_path).convert('RGB') # Load RGB image (ensure 3 channels even if the source is grayscale)

        if self.transform:
            image = self.transform(image)# Apply transforms if provided
        return image, label # Return tensor image and integer label

# -----------------------------
# Datasets & DataLoaders
# -----------------------------
# Note: train/val/test CSVs are generated from a clean split
# (Training/ used for train+val, original Testing/ kept as final test)
train_dataset = BrainMRIDataset('train.csv', transform=train_transform)
val_dataset = BrainMRIDataset('val.csv', transform=val_test_transform)
test_dataset = BrainMRIDataset('test.csv', transform=val_test_transform)

# Shuffling only for training; val/test stay deterministic
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
