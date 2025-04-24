import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# 超參數設定
DATA_DIR = "dataset/train"
BATCH_SIZE = 4
NUM_WORKERS = 2
LEARNING_RATE = 0.001
TEMPERATURE = 0.07
EPOCHS = 10
LOSSES_FILE = "training_losses.json"
MODEL_SAVE_PATH = "self_supervised_mobilenetv3.pth"
OUTPUT_DIR = "visualizations"
FEATURE_DIM = 128

# 設置設備
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# 資料集定義
class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.dataset = ImageFolder(root=root_dir, transform=None)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        x1 = self.transform(image)
        x2 = self.transform(image)
        return x1, x2, label

# 定義數據增強
contrast_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 自監督學習模型
class SimCLRNet(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
        # 使用 MobileNetV3 Small 作為基礎模型
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        
        self.projector = nn.Sequential(
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        h = self.backbone.features(x)
        h = F.adaptive_avg_pool2d(h, (1, 1))
        h = h.view(h.size(0), -1)
        # 添加特徵維度調試資訊 (第一次執行時可以取消這行的註解來檢查維度)
        # print(f"Feature shape: {h.shape}")
        z = self.projector(h)
        return F.normalize(z, dim=1)

# 獲取模型
def get_model():
    return SimCLRNet(feature_dim=FEATURE_DIM)

# 對比學習損失函數
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=0)
        N = z1.size(0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim = sim / self.temperature
        mask = ~torch.eye(2 * N, dtype=bool).to(device)
        sim = sim.masked_fill(~mask, -9e15)
        labels = torch.cat([torch.arange(N) + N, torch.arange(N)]).to(device)
        return F.cross_entropy(sim, labels)

# 自監督訓練函數
def train_self_supervised(model, data_loader, optimizer, criterion, device, epochs, save_path=None):
    losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x1, x2, _ in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # 獲取兩種增強版本的圖像
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            # 計算損失並反向傳播
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 記錄每個 epoch 的損失
        epoch_loss = running_loss / len(data_loader)
        losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

        # 每 2 個 epoch 儲存一次損失記錄或最後一個 epoch
        if save_path and (epoch % 2 == 0 or epoch == epochs - 1):
            with open(save_path, 'w') as f:
                json.dump(losses, f)
            print(f"損失記錄已儲存至: {save_path}")

    # 訓練結束後儲存最終損失記錄
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(losses, f)
        print(f"最終損失記錄已儲存至: {save_path}")

    return losses

# 主函數
def main():
    print(f"使用設備: {device}")
    
    # 加載數據集
    dataset = SimCLRDataset(root_dir=DATA_DIR, transform=contrast_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"數據集大小: {len(dataset)}")
    
    # 初始化模型、優化器與損失函數
    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = NTXentLoss(temperature=TEMPERATURE)

    # 確保輸出目錄存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 開始訓練
    print("開始訓練...")
    losses = train_self_supervised(model, dataloader, optimizer, criterion, device, epochs=EPOCHS, save_path=LOSSES_FILE)

    # 儲存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("訓練完成！模型已儲存")

if __name__ == "__main__":
    main()