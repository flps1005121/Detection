import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import datetime  # 導入 datetime 模組用於時間戳記
import sqlite3   # 添加SQLite數據庫支持

# 導入我們剛創建的特徵提取器
from feature_extractor import ImageFeatureExtractor

def log_to_file(log_file, message, print_to_console=True):
    """將訊息同時輸出到控制台和日誌檔案"""
    if print_to_console:
        print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

def load_features_from_database(db_file, selected_label=None):
    """從SQLite數據庫讀取特徵數據，可選擇性地根據標籤過濾"""
    if not os.path.exists(db_file):
        print(f"錯誤：特徵數據庫 {db_file} 不存在")
        return [], [], []
    
    try:
        # 連接到SQLite數據庫
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # 根據是否提供標籤來決定查詢
        if selected_label is not None:
            # 只讀取指定標籤的數據
            cursor.execute("SELECT file_path, label, feature FROM features WHERE label = ?", (selected_label,))
            print(f"正在讀取標籤為 '{selected_label}' 的特徵...")
        else:
            # 讀取所有數據
            cursor.execute("SELECT file_path, label, feature FROM features")
            print("讀取所有特徵...")
        
        rows = cursor.fetchall()
        
        if not rows:
            if selected_label:
                print(f"數據庫中沒有標籤為 '{selected_label}' 的特徵記錄")
            else:
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
        if selected_label:
            print(f"篩選條件: 標籤 = '{selected_label}'")
        return features, file_paths, labels
        
    except Exception as e:
        print(f"從數據庫讀取特徵時發生錯誤: {e}")
        return [], [], []

def predict_and_display(test_dir, db_file='train_features.db', log_file=None, selected_label=None):
    """預測並顯示結果，只保留結果摘要"""
    # 創建特徵提取器實例
    extractor = ImageFeatureExtractor(model_path=model_path)
    
    # 使用本地函數從SQLite數據庫載入特徵，可指定標籤
    try:
        train_features, train_file_paths, train_labels = load_features_from_database(db_file, selected_label)
        if len(train_features) == 0:
            if selected_label:
                log_to_file(log_file, f"錯誤：找不到標籤為 '{selected_label}' 的特徵資料")
            else:
                log_to_file(log_file, "錯誤：訓練集特徵資料庫為空")
            return
    except Exception as e:
        log_to_file(log_file, f"錯誤：無法載入特徵資料庫 {db_file}: {str(e)}")
        return

    # 使用與特徵提取器相同的轉換函數
    test_transform = extractor.transform
    
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 使用ImageFolder處理測試圖像
    results = []
    
    for idx, (img, class_idx) in enumerate(tqdm(test_loader, desc="處理圖像")):
        img_path = test_dataset.samples[idx][0]  # 獲取原始圖像路徑
        img_name = os.path.basename(img_path)
        class_name = test_dataset.classes[class_idx.item()]
        
        # 靜默處理每張圖像
        process_test_image(img_path, img_name, train_features, train_file_paths, train_labels, extractor, results, None, class_name)
    
    # 只顯示結果摘要
    display_results_summary(results, log_file)
    
    return results

def process_test_image(test_img_path, test_img_name, train_features, train_file_paths, train_labels, extractor, results, log_file=None, class_name=None):
    """靜默處理單個測試圖像並收集結果，不輸出中間過程"""
    # 提取特徵
    query_feature = extractor.extract_features(test_img_path)
    
    if query_feature is None:
        return
    
    # 計算相似度
    similarities = []
    for idx, train_feature in enumerate(train_features):
        # 余弦相似度計算
        sim = np.dot(query_feature, train_feature) / (np.linalg.norm(query_feature) * np.linalg.norm(train_feature))
        similarities.append((train_file_paths[idx], train_labels[idx], sim))
    
    # 根據相似度排序
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # 只處理最相似的結果
    if not similarities:
        return

    # 找出最相似的圖片
    most_similar_path, most_similar_label, highest_similarity = similarities[0]
    most_similar_name = os.path.basename(most_similar_path)

    # 保存結果以便摘要顯示
    results.append({
        'test_image': test_img_name,
        'class_name': class_name,
        'most_similar': most_similar_name,
        'most_similar_label': most_similar_label,
        'similarity': highest_similarity
    })

def display_results_summary(results, log_file=None):
    """顯示結果摘要，只輸出標籤與相似度"""
    log_to_file(log_file, "預測結果摘要:")
    log_to_file(log_file, "-"*50)
    for result in results:
        log_to_file(log_file, f"{result['test_image']} -> 標籤: {result['most_similar_label']}, 相似度: {result['similarity']:.4f}")

# 直接執行的主程式碼
if __name__ == "__main__":
    # 要預測的參數設定
    test_dir = 'feature_db/test/'
    model_path = 'output/simclr_mobilenetv3.pth'
    db_file = 'output/train_features.db'
    selected_label = 'vending'  # 預設為None，表示比對所有標籤
    
    # 可以在這裡設定要比對的特定標籤
    # selected_label = "特定標籤名稱"  # 例如: "cat", "dog" 等
    
    # 確保結果目錄存在
    result_dir = 'output/result/'
    os.makedirs(result_dir, exist_ok=True)
    
    # 使用時間戳記生成唯一的檔名
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label_info = f"_{selected_label}" if selected_label else ""
    log_file = f'{result_dir}prediction{label_info}_{timestamp}.txt'
    
    # 執行相似圖像預測
    print(f"\n執行相似圖像預測...\n結果將保存到: {log_file}")
    
    # 執行預測，移除了top_k參數
    predict_and_display(test_dir, db_file, log_file, selected_label)