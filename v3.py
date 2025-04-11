# SimCLR + MobileNetV2 + Prototype-based Few-shot Classification (Enhanced Version)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

# ------------------------
# CONFIGURATION
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
feature_dim = 128
num_epochs = 200 # 原本是 100
few_shot_k = 3

data_dir = "dataset/train"
test_dir = "dataset/test"

# ------------------------
# Dataset Definition
# ------------------------
class SimCLRDataset(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root=root)
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        x1 = self.transform(image)
        x2 = self.transform(image)
        return x1, x2, label

contrast_transforms = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.5, 0.5, 0.5, 0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = SimCLRDataset(root=data_dir, transform=contrast_transforms)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"Total training samples: {len(train_dataset)}")

# ------------------------
# SimCLR Network
# ------------------------
class SimCLRNet(nn.Module):
    def __init__(self, base_encoder, feature_dim):
        super().__init__()
        self.encoder = base_encoder
        self.projector = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = F.adaptive_avg_pool2d(h, (1, 1))
        h = h.view(h.size(0), -1)
        z = self.projector(h)
        return F.normalize(z, dim=1)

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
mobilenet.classifier = nn.Identity()
model = SimCLRNet(mobilenet.features, feature_dim).to(device)

# ------------------------
# NT-Xent Loss
# ------------------------
def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    N = z1.size(0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim = sim / temperature
    mask = ~torch.eye(2 * N, dtype=bool).to(device)
    sim = sim.masked_fill(~mask, -9e15)
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)]).to(device)
    loss = F.cross_entropy(sim, labels)
    return loss

# ------------------------
# Training
# ------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4) # 原本是5e-4 -> 1e-4 -> 3e-4 -> 2e-4
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# 記錄每個 epoch 的平均損失
losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x1, x2, _ in data_loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1, z2 = model(x1), model(x2)
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    #scheduler.step()
    avg_loss = total_loss / len(data_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader):.4f}")

# 繪製 Loss 圖
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------
# Few-shot Prototype Classification
# ------------------------
infer_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
infer_dataset = ImageFolder(root=data_dir, transform=infer_transform)
class_to_idx = infer_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
print("Train class_to_idx:", class_to_idx)

# Collect embeddings for prototype per class
model.eval()
class_to_embeddings = defaultdict(list)
for x, y in infer_dataset:
    emb = model(x.unsqueeze(0).to(device)).squeeze(0).detach().cpu()
    class_to_embeddings[y].append(emb)

# Split into support and query sets, compute prototypes
support_embeddings, support_labels = [], []
query_embeddings, query_labels = [], []

for class_id, embeddings in class_to_embeddings.items():
    print(f"Class {idx_to_class[class_id]} has {len(embeddings)} samples")
    random.shuffle(embeddings)
    support_embs = embeddings[:few_shot_k]  # 前 few_shot_k 張作為 support
    query_embs = embeddings[few_shot_k:]    # 剩餘作為 query
    support_embeddings.extend(support_embs)
    support_labels.extend([class_id] * len(support_embs))
    query_embeddings.extend(query_embs)
    query_labels.extend([class_id] * len(query_embs))

support_embeddings = torch.stack(support_embeddings)
support_labels = np.array(support_labels)
print(f"Support set size: {len(support_labels)} samples")

if query_embeddings:
    query_embeddings = torch.stack(query_embeddings)
    query_labels = np.array(query_labels)
    print(f"Query set size: {len(query_labels)} samples")
else:
    print("Warning: Query set is empty.")
    exit()

# Compute class prototypes from support set
support_prototypes = torch.stack([support_embeddings[support_labels == c].mean(dim=0) for c in range(len(class_to_idx))])

# Predict using prototype distance
preds = []
for i, q in enumerate(query_embeddings):
    dists = torch.norm(support_prototypes - q, dim=1)
    pred_class = torch.argmin(dists).item()
    preds.append(pred_class)
    print(f"Query {i}: True = {idx_to_class[query_labels[i]]}, Predicted = {idx_to_class[pred_class]}")

acc = np.mean(np.array(preds) == query_labels)
print(f"Few-shot prototype classification accuracy: {acc:.2f}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(query_labels, preds)
ConfusionMatrixDisplay(cm, display_labels=[idx_to_class[i] for i in range(len(idx_to_class))]).plot(cmap="Blues")
plt.title("Confusion Matrix (Query Set)")
plt.show()

# ------------------------
# t-SNE Visualization
# ------------------------
embeddings = torch.cat([support_prototypes, query_embeddings]).numpy()
labels = np.concatenate([np.arange(len(support_prototypes)), query_labels])
tsne = TSNE(n_components=2, perplexity=min(5, len(embeddings)-1), learning_rate=100)
reduced = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for i in range(len(class_to_idx)):
    idx = np.where(labels == i)
    plt.scatter(reduced[idx, 0], reduced[idx, 1], label=idx_to_class[i])
plt.legend()
plt.title("t-SNE of Embeddings")
plt.show()

# ------------------------
# Test Evaluation (with prototype)
# ------------------------
if os.path.exists(test_dir) and os.listdir(test_dir):
    test_dataset = ImageFolder(root=test_dir, transform=infer_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("Test class_to_idx:", test_dataset.class_to_idx)

    correct, total = 0, 0
    with torch.no_grad():
        for x, label in test_loader:
            emb = model(x.to(device)).squeeze(0).cpu()
            dists = torch.norm(support_prototypes - emb, dim=1)
            pred_class = torch.argmin(dists).item()
            print(f"GT: {idx_to_class.get(label.item(), 'unknown')}, Pred: {idx_to_class[pred_class]}")
            correct += (pred_class == label.item())
            total += 1
    print(f"Test accuracy: {correct / total:.2f}")
else:
    print(f"Test directory '{test_dir}' not found or empty.")


# ------------------------
# Auto experiment over different few_shot_k values
# ------------------------

max_k = min(len(v) for v in class_to_embeddings.values())  # 最小類別圖片數 = 最大 few_shot_k
with open("results_log.txt", "w") as f:
    f.write("few_shot_k, Train acc, Test acc\n")

print(f"\n[AutoEval] Running experiments for few_shot_k = 1 to {max_k}")

for k in range(1, max_k + 1):
    print(f"\n[AutoEval] --- Testing few_shot_k = {k} ---")
    support_embeddings, support_labels = [], []
    query_embeddings, query_labels = [], []

    for class_id, embeddings in class_to_embeddings.items():
        random.shuffle(embeddings)
        support_embs = embeddings[:k]
        query_embs = embeddings[k:]
        support_embeddings.extend(support_embs)
        support_labels.extend([class_id] * len(support_embs))
        query_embeddings.extend(query_embs)
        query_labels.extend([class_id] * len(query_embs))

    support_embeddings = torch.stack(support_embeddings)
    support_labels = np.array(support_labels)
    query_embeddings = torch.stack(query_embeddings)
    query_labels = np.array(query_labels)

    support_prototypes = torch.stack([
        support_embeddings[support_labels == c].mean(dim=0)
        for c in range(len(class_to_idx))
    ])

    # Query classification
    preds = []
    for q in query_embeddings:
        dists = torch.norm(support_prototypes - q, dim=1)
        pred_class = torch.argmin(dists).item()
        preds.append(pred_class)
    train_acc = np.mean(np.array(preds) == query_labels)

    # Test evaluation
    correct, total = 0, 0
    with torch.no_grad():
        for x, label in test_loader:
            emb = model(x.to(device)).squeeze(0).cpu()
            dists = torch.norm(support_prototypes - emb, dim=1)
            pred_class = torch.argmin(dists).item()
            correct += (pred_class == label.item())
            total += 1
    test_acc = correct / total

    print(f"→ Train acc = {train_acc:.2f}, Test acc = {test_acc:.2f}")

    with open("results_log.txt", "a") as f:
        f.write(f"few_shot_k={k}, Train acc={train_acc:.2f}, Test acc={test_acc:.2f}\n")


import pandas as pd

df = pd.read_csv("results_log.txt", skiprows=1, names=["k", "train_acc", "test_acc"])
plt.plot(df["k"], df["train_acc"], label="Train Acc", marker='o')
plt.plot(df["k"], df["test_acc"], label="Test Acc", marker='s')
plt.xlabel("few_shot_k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs few_shot_k")
plt.legend()
plt.grid(True)
plt.show()