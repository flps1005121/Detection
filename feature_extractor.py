import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import sqlite3
# 從model_train導入必要的功能
from model_train import SimCLRNet, MODEL_SAVE_PATH, device

# 定義推論用的圖像轉換
infer_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 特徵提取器類別
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
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
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
            from new_predict import load_features_from_database
            return load_features_from_database(db_file)
        
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
        
        for idx, (img_tensor, label) in enumerate(tqdm(dataloader)):
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                feature = self.model(img_tensor)
            
            # 獲取圖像的文件路徑和類別標籤
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

# 主函數
def main():
    print(f"使用設備: {device}")
    
    # 特徵提取測試並使用SQLite儲存
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
