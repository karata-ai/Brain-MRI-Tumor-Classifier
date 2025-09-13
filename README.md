# 🧠 Brain MRI Tumor Classifier

Deep learning project for **automatic brain tumor classification from MRI scans** using PyTorch.  
The model is trained to distinguish between four categories:  
- **Glioma**  
- **Meningioma**  
- **Pituitary tumor**  
- **No tumor**

---

## 📂 Project Structure
- `data_loader.py` – custom PyTorch `Dataset` & `DataLoader` for train/val/test splits  ( Note: The dataset is from Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- `model.py` – CNN architecture (5 convolutional blocks + fully connected head)  
- `train.py` – training loop with checkpointing, LR scheduler, and performance plots  
- `test.py` – final evaluation, confusion matrix, classification report, and Grad-CAM visualizations  
- `external_eval.py` – evaluation on external BRISC dataset for generalization testing  

---

## 📊 Results

### Internal Test (Nickparvar dataset, clean split)
- **Accuracy:** 98.0%  
- Balanced performance across all four classes  
          
### External Test (BRISC 2025 dataset)
- **Accuracy:** 93.7%  
- Demonstrates strong generalization to unseen MRI data from different scanners  

---

## 📊 Classification Report (Internal)

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Glioma      | 1.00      | 0.97   | 0.98     | 300     |
| Meningioma  | 0.96      | 0.95   | 0.96     | 306     |
| Notumor     | 0.97      | 1.00   | 0.99     | 405     |
| Pituitary   | 0.99      | 1.00   | 0.99     | 300     |
| **Accuracy**|           |        | **0.98** | **1311**|
| Macro Avg   | 0.98      | 0.98   | 0.98     | 1311    |
| Weighted Avg| 0.98      | 0.98   | 0.98     | 1311    |

---

## 📈 Performance Visualization
| Training Loss & Validation Accuracy | Confusion Matrix | Grad-CAM |
|-------------------------------------|------------------|----------|
| ![loss_acc](loss_accuracy.png) | ![cm](reports/confusion_matrix.png) | ![gradcam](gradcam_samples/gradcam_0_glioma_pred-glioma.png) |

---

## ⚙️ Technical Details
- **Framework:** PyTorch  
- **Input size:** 224×224 RGB MRI scans  
- **Architecture:**  
  - 5× Conv2D + BatchNorm + ReLU + MaxPool  
  - Fully connected classifier with dropout  
- **Training setup:**  
  - Optimizer: Adam (lr=0.001)  
  - Scheduler: ReduceLROnPlateau  
  - Loss: CrossEntropyLoss  
  - Epochs: 35  

---

## 📌 Key Features
- Clean train/val/test splits (no data leakage)  
- Explainability with **Grad-CAM** heatmaps  
- External validation for robust evaluation  
- Automatic saving of best model checkpoint  

---

## 🚀 Future Work
- Experiment with transfer learning (ResNet/EfficientNet)  
- Deploy as an inference API for real-time MRI classification  
- Explore 3D MRI volumes instead of 2D slices  
