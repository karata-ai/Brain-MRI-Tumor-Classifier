import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainTumorCNN(nn.Module):
    """
       Custom CNN for 4-class brain MRI classification (glioma, meningioma, notumor, pituitary).

       Design notes:
       - 5 convolutional blocks with BatchNorm + ReLU + MaxPool(2) each.
       - With input 224x224, five 2x downsamplings yield a 7x7 spatial map.
       - Classifier: Flatten -> Linear(512*7*7 -> 256) -> Dropout -> Linear(256 -> 4).

       Tip: If you later change input size or pooling depth, consider replacing the flatten
       with an AdaptiveAvgPool2d to (1,1) and a Linear(512 -> 256) for shape-robustness.
       """
    def __init__(self):
        super(BrainTumorCNN, self).__init__()

        # ----------------------------
        # Feature extractor
        # ----------------------------
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # preserve HxW
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 224 -> 112

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 112 -> 56

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 56 -> 28

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 28 -> 14

            # Block 5: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 14 -> 7
        )

        # ----------------------------
        # Classifier head
        # ----------------------------
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),    # flatten 512x7x7 -> 256
            nn.ReLU(),
            nn.Dropout(0.5),                        # regularize fully-connected layer
            nn.Linear(256, 4)  # 4 target classes
        )

    def forward(self, x):
        # Extract convolutional features
        x = self.features(x)

        # Flatten to (B, 512*7*7). If input size changes, this shape must match.
        x = x.view(x.size(0), -1)

        # Class logits (unnormalized). Apply CrossEntropyLoss downstream.
        x = self.classifier(x)
        return x