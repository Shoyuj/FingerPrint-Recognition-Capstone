# Biometric Siamese Neural Network for Fingerprint Recognition

An end-to-end Deep Metric Learning system using Siamese Convolutional Neural Networks (CNNs) for robust biometric fingerprint verification (1:1) and identification (1:N), trained on the **Sokoto Coventry Fingerprint Dataset (SOCOFing)**.

---

## Key Features

- **Siamese CNN Architecture**: Twin feature extractors with weight-sharing, L1 feature distance subtraction, and dense classification layers.
- **Biometric 1:1 Verification**: Evaluates whether two fingerprint impressions belong to the same finger identity with a calibrated match probability.
- **Biometric 1:N Identification**: Rapidly searches an unknown or distorted probe fingerprint across 6,000 gallery fingerprints to identify the subject.
- **Robust Data Augmentation**: Real-time Gaussian blur, rotation (-30° to +30°), scaling (90% to 110%), translation, and noise simulation to handle imperfect fingerprint scans.
- **Interactive Web UI**: Modern dark-mode glassmorphic web application with live scanning effects, distortion playground, and gallery browser.
- **Modern Keras 3 & Python 3.9+ Support**: Fully compatible with TensorFlow 2.x and Keras 3.

---

## Quick Start

### 1. Launch the Interactive Web App

Run the runner script (or start `app.py` directly):

```bash
# Start web app on port 8080
./run.sh app
```

Open **`http://localhost:8080`** in your browser.

### 2. Run 1:1 Verification CLI Demo

```bash
./run.sh verify
```

This selects random query fingerprints, compares them against true database matches and imposters, and saves a side-by-side visualization to `result/verification_result.png`.

### 3. Run 1:N Identification Search

```bash
./run.sh search
```

Searches altered probe fingerprints across the gallery of 6,000 real fingerprints and outputs Top-5 ranked candidates.

### 4. Evaluate Test Set Metrics

```bash
./run.sh eval
```

Computes Test Accuracy, Precision, Recall, F1 Score, True Positive, and False Positive rates on balanced test pairs.

### 5. Train the Siamese Model

```bash
# Full training (8 epochs on all 49,270 altered samples)
./run.sh train 8 64

# Quick training (3 epochs on 10,000 samples)
./run.sh quick-train
```

---

## Project Structure

```
fingerprint_recognition/
├── dataset/                    # Preprocessed SOCOFing datasets
│   ├── x_real.npz / y_real.npy # 6,000 real gallery fingerprints
│   ├── x_easy.npz / y_easy.npy # 17,931 Easy altered fingerprints
│   ├── x_medium.npz / y_medium.npy # 17,067 Medium altered fingerprints
│   └── x_hard.npz / y_hard.npy # 14,272 Hard altered fingerprints
├── web/                        # Web application frontend assets
│   ├── index.html              # Modern single-page UI template
│   ├── style.css               # Glassmorphic dark-mode styles & animations
│   └── app.js                  # Interactive client-side logic
├── model.py                    # Siamese CNN architecture definition
├── data_generator.py           # Online pair generator & augmentations
├── train.py                    # Production model training script
├── recognize.py                # Verification, DB search & evaluation CLI
├── app.py                      # REST API & Web UI server
├── run.sh                      # Unified execution helper
├── requirements.txt            # Python dependencies
├── train.ipynb                 # Modernized training notebook
├── preprocess.ipynb            # Dataset preprocessing notebook
└── result/                     # Model weights & evaluation plots
    ├── fingerprint_model.keras # Trained Siamese CNN weights
    ├── training_curves.png     # Loss and accuracy curves
    └── verification_result.png # Sample verification visual outputs
```

---

## Biometric Label Encoding Format

Each fingerprint sample is labeled with 4 attributes: `[subject_id, gender, lr, finger]`
e.g., `[122, 0, 0, 1]` represents:
- **Subject ID:** `122` (0 to 599)
- **Gender:** `0` = Male, `1` = Female
- **Hand:** `0` = Left Hand, `1` = Right Hand
- **Finger:** `0` = Thumb, `1` = Index, `2` = Middle, `3` = Ring, `4` = Little

---

## Model Architecture

```
Input Pair (x1, x2) [Shape: (90, 90, 1)]
       │
       ▼
┌─────────────────────────────────────────┐
│ Shared Feature Extractor CNN (W_theta)  │
│  - Conv2D(32, 3x3, ReLU) + MaxPool(2x2) │
│  - Conv2D(32, 3x3, ReLU) + MaxPool(2x2) │
└─────────────────────────────────────────┘
       │
       ▼
 Feature Subtraction: Diff = | W(x1) - W(x2) |
       │
       ▼
┌─────────────────────────────────────────┐
│ Metric Classification Head              │
│  - Conv2D(32, 3x3, ReLU) + MaxPool(2x2) │
│  - Flatten()                            │
│  - Dense(64, ReLU)                      │
│  - Dense(1, Sigmoid)                    │
└─────────────────────────────────────────┘
       │
       ▼
 Match Confidence Probability [0.0 - 1.0]
```

---

## Dataset Reference

- **SOCOFing (Sokoto Coventry Fingerprint Dataset)**: [Kaggle Dataset](https://www.kaggle.com/ruizgara/socofing/home)
