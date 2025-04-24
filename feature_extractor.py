import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

# Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
feature_dim = 128
data_dir = "dataset/train"
test_dir = "dataset/test"

# Logging utility
log_file = "log.txt"
if not os.path.exists(log_file):
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("")  # 初始化 log file

def log_print(*args, **kwargs):
    message = ' '.join(map(str, args))
    print(message, **kwargs)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# 自定義資料集，回傳兩個不同增強版本的同一張圖片（SimCLR用）
class SimCLRDataset(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root=root)
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        x1 = self.transform(image)  # 第一次增強
        x2 = self.transform(image)  # 第二次增強
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
log_print(f"Total training samples: {len(train_dataset)}")


# SimCLR 模型定義
class SimCLRNet(nn.Module):
    def __init__(self, base_encoder, feature_dim):
        super().__init__()
        self.encoder = base_encoder
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = self.encoder(dummy)
        self.output_dim = out.shape[1]
        self.projector = nn.Sequential(
            nn.Linear(self.output_dim, 512),  # 第一層全連接層
            nn.BatchNorm1d(512),             # 批次正規化，穩定訓練
            nn.ReLU(),                       # 激活函數
            nn.Dropout(0.2),                 # Dropout 防止過擬合
            nn.Linear(512, 512),             # 第二層全連接層
            nn.BatchNorm1d(512),             # 批次正規化
            nn.ReLU(),                       # 激活函數
            nn.Dropout(0.2),                 # Dropout
            nn.Linear(512, feature_dim)      # 最終映射到指定特徵維度
        )

    def forward(self, x):
        h = self.encoder(x)  # 提取 feature map
        h = F.adaptive_avg_pool2d(h, (1, 1))  # 全域平均池化
        h = h.view(h.size(0), -1)  # 展平
        z = self.projector(h)  # 投影到低維度空間
        return F.normalize(z, dim=1)  # 單位向量化，方便算 cosine similarity

def get_model():
    """返回用於訓練的 SimCLRNet 模型"""
    mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    mobilenet.classifier = nn.Identity()
    return SimCLRNet(mobilenet.features, feature_dim).to(device)

# 特徵提取器，支援預處理、模型加載、特徵儲存與搜尋
class ImageFeatureExtractor:
    def __init__(self, model_path='self_supervised_mobilenetv3.pth', device=None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        log_print(f"Using device: {self.device}")

        # 創建模型並載入預訓練權重
        # MobileNetV3 去掉 classifier 後使用 features 作為 backbone
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        mobilenet.classifier = nn.Identity()
        self.model = SimCLRNet(mobilenet.features, feature_dim).to(self.device)

        # 嘗試載入模型權重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            log_print(f"成功載入模型權重 {model_path}")
        except Exception as e:
            log_print(f"載入模型權重失敗: {e}")
            log_print("將使用未訓練的模型進行特徵提取")

        self.model.eval()

        # 圖像預處理
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def extract_features(self, image_path):
        """從圖像路徑提取特徵向量，回傳一張圖的特徵向量（flatten）"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 使用模型的backbone提取特徵
                features = self.model(image_tensor)

            return features.cpu().numpy().flatten()
        except Exception as e:
            log_print(f"處理圖片 {image_path} 時發生錯誤: {e}")
            return None

    def create_features_database(self, train_dir, cache_file='train_features.npz', force_refresh=True):
        """為訓練集中所有圖像創建特徵資料庫，把訓練集每張圖都跑過 extract_features，儲存起來作為搜尋資料庫"""
        if not os.path.exists(train_dir):
            print(f"錯誤：指定的目錄 {train_dir} 不存在")
            return [], []

        image_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)
                       if os.path.isfile(os.path.join(train_dir, f)) and
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))]

        if not image_files:
            log_print(f"警告：在目錄 {train_dir} 中未找到任何圖像檔案")
            return [], []

        log_print(f"正在為 {len(image_files)} 張訓練圖像提取特徵...")
        features = []
        valid_file_paths = []

        for img_path in tqdm(image_files):
            feature = self.extract_features(img_path)
            if feature is not None:
                features.append(feature)
                valid_file_paths.append(img_path)

        if not features:
            log_print("警告：沒有成功從任何圖像中提取特徵")
            return [], []

        try:
            # 保存特徵資料庫
            np.savez(cache_file, features=np.array(features), file_paths=np.array(valid_file_paths))
            log_print(f"特徵資料庫已儲存到 {os.path.abspath(cache_file)}")
        except Exception as e:
            log_print(f"儲存特徵資料庫時發生錯誤: {e}")

        return features, valid_file_paths

    def find_similar_images(self, query_img_path, features, file_paths, top_k=5):
        """尋找與查詢圖像最相似的訓練圖像，用 cosine similarity 找出最相近的圖像"""
        query_feature = self.extract_features(query_img_path)

        if query_feature is None:
            return []

        # 計算餘弦相似度
        similarities = []
        for idx, feat in enumerate(features):
            sim = np.dot(query_feature, feat) / (np.linalg.norm(query_feature) * np.linalg.norm(feat))
            similarities.append((file_paths[idx], sim))

        # 根據相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

# 保留直接執行特徵提取的程式碼以供測試
if __name__ == "__main__":
    train_dir = 'dataset/train'  # 訓練圖像資料夾路徑
    model_path = 'self_supervised_mobilenetv3.pth'  # 模型權重檔案路徑
    cache_file = 'train_features.npz'  # 特徵資料庫儲存路徑

    print("初始化特徵提取器...")
    extractor = ImageFeatureExtractor(model_path=model_path)

    print(f"\n開始從 {train_dir} 提取特徵...")
    features, file_paths = extractor.create_features_database(
        train_dir=train_dir,
        cache_file=cache_file,
        force_refresh=True
    )
    print(f"\n成功提取了 {len(features)} 張圖像的特徵")
