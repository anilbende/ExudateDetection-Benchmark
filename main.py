#!/usr/bin/env python3
"""
main.py - Unified entry point for ExudateDetection-Benchmark
Includes preprocessing, classical ML, CNN, and evaluation in one file.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
from skimage import io, transform, exposure, color
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt

# -----------------------------
# Preprocessing
# -----------------------------
def load_images(data_dir, target_size=(224, 224)):
    images, labels = [], []
    for fname in os.listdir(data_dir):
        if fname.endswith(".jpg") or fname.endswith(".png"):
            img = io.imread(os.path.join(data_dir, fname))
            img_resized = transform.resize(img, target_size, anti_aliasing=True)
            images.append(img_resized)
            labels.append(1 if "exudate" in fname.lower() else 0)
    return np.array(images), np.array(labels)

def augment_image(img):
    angle = np.random.uniform(-15, 15)
    rotated = transform.rotate(img, angle)
    if np.random.rand() > 0.5:
        rotated = np.fliplr(rotated)
    gamma = np.random.uniform(0.8, 1.2)
    return exposure.adjust_gamma(rotated, gamma)

def preprocess_data(data_dir, test_size=0.2, target_size=(224, 224), augment=False):
    images, labels = load_images(data_dir, target_size)
    if augment:
        aug_images = [augment_image(img) for img in images]
        images = np.concatenate([images, aug_images])
        labels = np.concatenate([labels, labels])
    return train_test_split(images, labels, test_size=test_size, stratify=labels, random_state=42)

# -----------------------------
# Classical ML Pipeline
# -----------------------------
def extract_features(images):
    feature_list = []
    for img in images:
        gray = color.rgb2gray(img)
        hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       orientations=9, block_norm="L2-Hys", feature_vector=True)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
        color_hist = []
        for i in range(3):
            hist, _ = np.histogram(img[:, :, i], bins=16, range=(0, 1), density=True)
            color_hist.extend(hist)
        features = np.concatenate([hog_feat, lbp_hist, color_hist])
        feature_list.append(features)
    return np.array(feature_list)

def run_classical_ml(X_train, X_test, y_train, y_test):
    X_train_feat = extract_features(X_train)
    X_test_feat = extract_features(X_test)
    predictions = {}
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_feat, y_train)
    predictions["logistic_regression"] = lr.predict_proba(X_test_feat)[:, 1]
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train_feat, y_train)
    predictions["svm_rbf"] = svm.predict_proba(X_test_feat)[:, 1]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_feat, y_train)
    predictions["random_forest"] = rf.predict_proba(X_test_feat)[:, 1]
    return predictions

# -----------------------------
# CNN Pipeline
# -----------------------------
def get_dataloaders(X_train, X_test, y_train, y_test, batch_size=16):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    X_train_t = torch.stack([transform(img.astype(np.float32)) for img in X_train])
    X_test_t = torch.stack([transform(img.astype(np.float32)) for img in X_test])
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def build_model(num_classes=2):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    return model

def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"[Epoch {epoch+1}] Training complete")
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds.extend(probs.cpu().numpy())
    return np.array(preds)

def run_cnn(X_train, X_test, y_train, y_test):
    train_loader, test_loader = get_dataloaders(X_train, X_test, y_train, y_test)
    model = build_model()
    preds = train_model(model, train_loader, test_loader, epochs=10,
                        device="cuda" if torch.cuda.is_available() else "cpu")
    return preds

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_results(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_probs),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

def plot_roc(y_true, y_pred_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc = roc_auc_score(y_true, y_pred_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

# -----------------------------
# Main Runner
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="ExudateDetection Benchmark")
    parser.add_argument("--method", choices=["classical", "cnn", "all"], required=True)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--results_dir", type=str, default="results/")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = preprocess_data(args.data_dir)

    results = {}
    if args.method in ["classical", "all"]:
        preds_dict = run_classical_ml(X_train, X_test, y_train, y_test)
        for clf, preds in preds_dict.items():
            metrics = evaluate_results(y_test, preds)
            results[clf] = metrics
            plot_roc(y_test, preds, os.path.join(args.results_dir, f"{clf}_roc.png"))

    if args.method in ["cnn", "all"]:
        preds = run_cnn(X_train, X_test, y_train, y_test)
        metrics = evaluate_results(y_test, preds)
        results["cnn"] = metrics
        plot_roc(y_test, preds, os.path.join(args.results_dir, "cnn_roc.png"))

    print("\n=== Results Summary ===")
    for method, metrics in results.items():
        print(f"{method.upper()}:")
        for k, v in metrics.items():
            print
