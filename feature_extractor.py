import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import sqlite3  # 新增SQLite模組
# 從model_train導入必要的功能
from model_train import SimCLRNet, MODEL_SAVE_PATH

# 設置設備
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# 定義推論用的圖像轉換
infer_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 新增 - 特徵提取器類別
class ImageFeatureExtractor:
    def __init__(self, model_path=MODEL_SAVE_PATH, device=None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                        "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"使用設備: {self.device}")
        
        # 創建模型並載入預訓練權重
        self.model = SimCLRNet()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"成功載入模型權重 {model_path}")
        except Exception as e:
            print(f"載入模型權重失敗: {e}")
            print("將使用未訓練的模型進行特徵提取")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 使用獨立出來的轉換變數
        self.transform = infer_transform
    
    def extract_features(self, image_path):
        """從圖像路徑提取特徵向量"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # 修正：使用模型的backbone提取特徵
                features = self.model.backbone(image_tensor)
                # 正規化特徵向量
                features = F.normalize(features, dim=1)
                
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"處理圖片 {image_path} 時發生錯誤: {e}")
            return None
    
    def create_features_database(self, train_dir, db_file='train_features.db', force_refresh=True):
        """為訓練集中所有圖像創建特徵資料庫，支持子類別目錄結構，並儲存到SQLite數據庫"""
        if not os.path.exists(train_dir):
            print(f"錯誤：指定的目錄 {train_dir} 不存在")
            return [], [], []
        
        # 檢查是否需要強制刷新數據庫
        if os.path.exists(db_file) and not force_refresh:
            print(f"發現已存在的數據庫 {db_file}，正在載入...")
            return self.load_features_from_database(db_file)
        
        # 使用ImageFolder讀取數據集
        dataset = ImageFolder(
            root=train_dir,
            transform=self.transform
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        if len(dataset) == 0:
            print(f"警告：在目錄 {train_dir} 中未找到任何有效的圖像檔案")
            return [], [], []
        
        print(f"正在為 {len(dataset)} 張訓練圖像提取特徵...")
        features = []
        valid_file_paths = []
        labels = []
        
        for img_tensor, label in tqdm(dataloader):
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                # 使用模型的backbone提取特徵
                feature = self.model.backbone(img_tensor)
                # 正規化特徵向量
                feature = F.normalize(feature, dim=1)
            
            # 獲取圖像的文件路徑和類別標籤
            idx = len(features)
            img_path = dataset.imgs[idx][0]
            class_name = dataset.classes[label.item()]
            
            features.append(feature.cpu().numpy().flatten())
            valid_file_paths.append(img_path)
            labels.append(class_name)
        
        if len(features) == 0:
            print(f"警告：沒有成功從任何圖像中提取特徵")
            return [], [], []
            
        try:
            # 建立SQLite數據庫連接
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # 創建特徵資料表
            cursor.execute('''CREATE TABLE IF NOT EXISTS features
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT,
                        label TEXT,
                        feature BLOB)''')
            
            # 如果需要刷新，先清空表
            if force_refresh:
                cursor.execute("DELETE FROM features")
                
            # 插入數據
            for file_path, label, feature in zip(valid_file_paths, labels, features):
                cursor.execute("INSERT INTO features (file_path, label, feature) VALUES (?, ?, ?)",
                              (file_path, label, feature.tobytes()))
                              
            # 提交更改並關閉連接
            conn.commit()
            conn.close()
            
            print(f"特徵資料庫已儲存到 {os.path.abspath(db_file)}")
            print(f"資料庫包含 {len(dataset.classes)} 個類別: {dataset.classes}")
        except Exception as e:
            print(f"儲存特徵資料庫時發生錯誤: {e}")
        
        return features, valid_file_paths, labels
    
    def load_features_from_database(self, db_file):
        """從SQLite數據庫讀取特徵數據"""
        if not os.path.exists(db_file):
            print(f"錯誤：特徵數據庫 {db_file} 不存在")
            return [], [], []
        
        try:
            # 連接到SQLite數據庫
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # 讀取所有數據
            cursor.execute("SELECT file_path, label, feature FROM features")
            rows = cursor.fetchall()
            
            if not rows:
                print("數據庫中沒有特徵記錄")
                return [], [], []
                
            file_paths = []
            labels = []
            features = []
            
            for file_path, label, feature_bytes in rows:
                # 將二進制數據轉換回numpy數組
                feature = np.frombuffer(feature_bytes, dtype=np.float32)
                
                file_paths.append(file_path)
                labels.append(label)
                features.append(feature)
                
            conn.close()
            
            print(f"已從 {db_file} 中載入 {len(features)} 個特徵記錄")
            return features, file_paths, labels
            
        except Exception as e:
            print(f"從數據庫讀取特徵時發生錯誤: {e}")
            return [], [], []
    
    def find_similar_images(self, query_img_path, db_file=None, features=None, file_paths=None, top_k=5):
        """尋找與查詢圖像最相似的訓練圖像，支持從數據庫或內存中的特徵進行搜索"""
        query_feature = self.extract_features(query_img_path)
        
        if query_feature is None:
            return []
        
        # 如果提供了數據庫文件但沒有提供特徵，則從數據庫讀取
        if db_file and (features is None or file_paths is None):
            features, file_paths, _ = self.load_features_from_database(db_file)
        
        if not features or not file_paths:
            print("沒有特徵數據可供比較")
            return []
        
        # 計算餘弦相似度
        similarities = []
        for idx, feat in enumerate(features):
            sim = np.dot(query_feature, feat) / (np.linalg.norm(query_feature) * np.linalg.norm(feat))
            similarities.append((file_paths[idx], sim))
        
        # 根據相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

# 主函數
def main():
    print(f"使用設備: {device}")
    
    # 修改 - 特徵提取測試並使用SQLite儲存
    print("\n開始測試特徵提取功能...")
    extractor = ImageFeatureExtractor(model_path=MODEL_SAVE_PATH)
    features, file_paths, labels = extractor.create_features_database(
        train_dir="feature_db_netwk/train/", 
        db_file='output/train_features.db',
        force_refresh=True
    )
    print(f"成功提取了 {len(features)} 張圖像的特徵")
    
    # 如果有特徵，顯示一些統計信息
    if features:
        unique_labels = set(labels)
        print(f"資料集中包含 {len(unique_labels)} 個類別")
        for label in unique_labels:
            count = labels.count(label)
            print(f"  - {label}: {count} 張圖像")

if __name__ == "__main__":
    main()
