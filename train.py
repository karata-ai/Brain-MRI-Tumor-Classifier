import torch
import torch.nn as nn
import torch.optim as optim
from numpy.f2py.crackfortran import verbose
from sympy import factor

from model import BrainTumorCNN
from data_loader import train_loader, val_loader
import matplotlib.pyplot as plt

# -----------------------------
# Training config
# -----------------------------
EPOCHS = 35
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model / loss / optimizer
model = BrainTumorCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()   # expects raw logits and integer labels
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Reduce LR when validation metric plateaus (robust and hands-free scheduling)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',     # we maximize validation accuracy
    factor=0.5,     # halve LR on plateau
    patience=3      # wait 3 epochs without improvement
)

train_losses = []
val_accuracies = []
best_val_acc = 0.0  # track best model for checkpointing

for epoch in range(EPOCHS):
    # -----------------------------
    # Train
    # -----------------------------
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)             # logits (B, 4)
        loss = criterion(outputs, labels)   # cross-entropy over classes
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # -----------------------------
    # Validate (no grad, eval mode)
    # -----------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)    # argmax over class logits
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)

    # Step the scheduler with the validation metric (accuracy)
    scheduler.step(val_acc)

    print(f'Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}%')

    # -----------------------------
    # Save best checkpoint
    # -----------------------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_brain_tumor_cnn.pth')
        print(f'New best model saved at epoch {epoch+1} with Val Accuracy:{val_acc:.2f}%')

# -----------------------------
# Curves (for analysis)
# -----------------------------
plt.plot(train_losses, label = 'Train Loss')
plt.plot(val_accuracies, label = 'Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training loss and Validation Accuracy')
plt.grid(True)
plt.legend()
plt.show()
