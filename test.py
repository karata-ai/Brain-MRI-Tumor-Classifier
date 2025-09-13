import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from model import BrainTumorCNN
from data_loader import test_loader # uses deterministic val_test_transform
from PIL import Image
import os

# -----------------------------
# Load best checkpoint & device
# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BrainTumorCNN().to(DEVICE)
model.load_state_dict(torch.load('best_brain_tumor_cnn.pth', map_location=DEVICE))
model.eval()    # disable dropout/bn updates for eval

# -----------------------------
# Class name resolution
# -----------------------------
# test_loader.dataset.label_map is {class_name: idx}; invert it to get idx -> name
inv_label_map = {v: k for k, v in test_loader.dataset.label_map.items()}
class_names = [inv_label_map[i] for i in range(len(inv_label_map))]

# -----------------------------
# Inference on test set
# -----------------------------
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)                  # (B, 4) raw scores
        preds = logits.argmax(dim=1)            # predicted class indices
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
test_acc = (y_true == y_pred).mean() * 100.0
print(f"Test Accuracy: {test_acc:.2f}%")

# -----------------------------
# Classification report (TXT + CSV)
# -----------------------------
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_text = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n", report_text)

os.makedirs("reports", exist_ok=True)
with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
    f.write(report_text)

pd.DataFrame(report_dict).to_csv("reports/per_class_metrics.csv", index=True)

# -----------------------------
# Confusion matrix (PNG)
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(cm, interpolation='nearest')  # default colormap
ax.set_title("Confusion Matrix")
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)

# annotate counts in each cell
th = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > th else "black")
fig.tight_layout()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("reports/confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Grad-CAM
# -----------------------------
# Simple Grad-CAM over a few samples to visualize model attention


def find_last_conv_module(net):
    # Locate the last Conv2d layer to hook activations/gradients for CAM.
    last = None
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

last_conv = find_last_conv_module(model)

class GradCAM:
    # Vanilla Grad-CAM implementation for CNN classifiers.
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        # register hooks
        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()         # (B, C, H, W)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()   # (B, C, H, W)

    def generate(self, input_tensor, class_idx=None):
        """
        Compute Grad-CAM heatmap for a single input.
        input_tensor: (1, C, H, W)
        class_idx: optional target class (defaults to argmax)
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        # Global average pooling върху градиентите
        grads = self.gradients  # (1, C, H, W)
        acts = self.activations # (1, C, H, W)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # normalize 0..1 and resize back to input spatial size
        cam -= cam.min()
        cam += 1e-8
        cam /= cam.max()

        # Преоразмеряване към оригиналния вход
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        return cam.squeeze(0).squeeze(0).cpu().numpy(), class_idx

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def tensor_to_pil(t):
    # De-normalize (mean=std=0.5) and convert a CHW tensor to a PIL image.
    t = t.detach().cpu()
    t = t * 0.5 + 0.5
    t = t.clamp(0, 1)
    npimg = (t.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(npimg)

os.makedirs("gradcam_samples", exist_ok=True)
cam_engine = GradCAM(model, last_conv)

# Save CAM overlays for first ~8 test images
saved = 0
for images, labels in test_loader:
    for i in range(images.size(0)):
        if saved >= 8:
            break
        img = images[i:i+1].to(DEVICE)   # (1, C, H, W)
        heatmap, pred_idx = cam_engine.generate(img, class_idx=None)

        # prepare base image (denormalized) and overlay CAM
        pil_img = tensor_to_pil(images[i])
        w, h = pil_img.size


        heat = (heatmap * 255).astype(np.uint8)
        heat_img = Image.fromarray(heat).resize((w, h), Image.BILINEAR)
        heat_img = heat_img.convert("RGBA")

        # simple monochrome overlay with alpha channel
        alpha = 128
        heat_colored = Image.merge("RGBA", (heat_img.split()[0], Image.new("L", (w, h), 0), Image.new("L", (w, h), 0), Image.new("L", (w, h), alpha)))

        base_rgb = pil_img.convert("RGBA")
        blended = Image.alpha_composite(base_rgb, heat_colored).convert("RGB")

        true_lab = class_names[labels[i].item()]
        pred_lab = class_names[pred_idx]
        out_path = f"gradcam_samples/gradcam_{saved}_{true_lab}_pred-{pred_lab}.png"
        blended.save(out_path)
        saved += 1
    if saved >= 8:
        break

cam_engine.close()

print(" Saved:")
print(" - reports/confusion_matrix.png")
print(" - reports/classification_report.txt")
print(" - reports/per_class_metrics.csv")
print(" - gradcam_samples/gradcam_*.png")