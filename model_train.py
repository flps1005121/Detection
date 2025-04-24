import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from feature_extractor import get_model, SimCLRDataset, contrast_transforms, log_print, device

# 超參數設定
DATA_DIR = "dataset/train"
BATCH_SIZE = 4
NUM_WORKERS = 2
LEARNING_RATE = 0.001
TEMPERATURE = 0.07
EPOCHS = 100
LOSSES_FILE = "training_losses.json"
MODEL_SAVE_PATH = "self_supervised_mobilenetv3.pth"
OUTPUT_DIR = "visualizations"

# 定義對比學習的數據增強
class ContrastiveTransformations:
    def __init__(self):
        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.weak_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return [self.strong_transform(x), self.weak_transform(x)]


# 自監督學習數據集
class SelfSupervisedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # 加載目錄中的所有圖片
        self.image_files = [(os.path.join(root_dir, f), f) for f in os.listdir(root_dir)
                            if os.path.isfile(os.path.join(root_dir, f)) and
                            f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))]
        self.transform = transform
        print(f"找到 {len(self.image_files)} 張圖片")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 讀取圖片並應用增強
        img_path, _ = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image) if self.transform else image

# 自監督學習模型
class SelfSupervisedModel(nn.Module):
    def __init__(self, base_model='mobilenet_v3_small', projection_dim=64):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        # 凍結 backbone 參數（這邊是可選的）
        #for param in self.backbone.parameters():
        #    param.requires_grad = False

        feature_dim = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x):
        return self.projection_head(self.backbone(x))

# 對比學習損失函數
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # 歸一化嵌入向量
        z_i, z_j = nn.functional.normalize(z_i, dim=1), nn.functional.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.matmul(representations, representations.t()) / self.temperature
        mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=z_i.device)
        pos_mask = torch.cat([torch.arange(len(z_i), device=z_i.device),
                            torch.arange(len(z_i), device=z_i.device)])
        positives = sim_matrix[torch.arange(len(pos_mask)), pos_mask]
        negatives = sim_matrix[~mask].view(len(pos_mask), -1)
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        # 計算交叉熵損失
        return nn.CrossEntropyLoss()(logits, torch.zeros(len(pos_mask), dtype=torch.long, device=z_i.device))


# 自監督訓練函數
def train_self_supervised(model, data_loader, optimizer, criterion, device, epochs, save_path):
    losses = []
    best_loss = float("inf")
    patience = 10
    counter = 0

    log_print(f"開始訓練，使用設備: {device}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x1, x2 in data_loader:
            # 獲取兩種增強版本的圖像
            x1, x2 = x1.to(device), x2.to(device)
            # 前向傳播
            z1, z2 = model(x1, x2)
            # 計算損失
            loss = criterion(z1, z2)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 記錄每個 epoch 的損失
        epoch_loss = running_loss / len(data_loader)
        losses.append(epoch_loss)
        log_print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

        # 早停檢查
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
            # 可選：儲存最佳模型
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            log_print(f"儲存最佳模型至: {os.path.join(OUTPUT_DIR, 'best_model.pth')}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping！在第 {epoch+1} epoch 模型不再進步")
                break

        # 每 2 個 epoch 儲存一次損失記錄
        if save_path and (epoch % 2 == 0 or epoch == epochs - 1):
            with open(save_path, 'w') as f:
                json.dump(losses, f)
            print(f"損失記錄已儲存至: {save_path}")

    # 保存最終模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    log_print(f"最終模型已儲存至: {MODEL_SAVE_PATH}")

    return losses

# 主函數
def main():
    # 設置設備
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    # 設置數據路徑與增強
    data_dir = DATA_DIR
    transform = ContrastiveTransformations()
    # 加載數據集
    dataset = SelfSupervisedDataset(root_dir=DATA_DIR, transform=contrast_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"數據集大小: {len(dataset)}")
    # 初始化模型、優化器與損失函數
    model = SelfSupervisedModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = NTXentLoss(temperature=TEMPERATURE)

    # 確保輸出目錄存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 開始訓練
    log_print("開始訓練...")
    losses = train_self_supervised(model, dataloader, optimizer, criterion, device, epochs=EPOCHS, save_path=LOSSES_FILE)

    # 儲存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("訓練完成！模型已儲存")

if __name__ == "__main__":
    main()
