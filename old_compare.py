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

def load_features_from_database(db_file):
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

def predict_and_display(test_dir, top_k=5, db_file='train_features.db', log_file=None):
    """預測並顯示結果，可選擇性地記錄到檔案中"""
    # 記錄開始時間
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(log_file, f"\n========== 預測開始 {timestamp} ==========\n")
    
    # 創建特徵提取器實例
    extractor = ImageFeatureExtractor(model_path=model_path)
    
    # 使用本地函數從SQLite數據庫載入特徵
    try:
        train_features, train_file_paths, train_labels = load_features_from_database(db_file)
        log_to_file(log_file, f"成功從 {db_file} 載入 {len(train_file_paths)} 個特徵向量")
    except Exception as e:
        log_to_file(log_file, f"無法載入特徵資料庫 {db_file}: {str(e)}")
        return

    if len(train_features) == 0:
        log_to_file(log_file, "無法繼續：訓練集特徵資料庫為空")
        return

    # 使用與特徵提取器相同的轉換函數，確保一致性
    # 讓我們直接使用特徵提取器的轉換函數
    test_transform = extractor.transform
    
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    log_to_file(log_file, f"成功載入測試集: {len(test_dataset)} 張圖像, {len(test_dataset.classes)} 個類別")
    log_to_file(log_file, f"類別映射: {test_dataset.class_to_idx}")
    
    # 使用ImageFolder處理測試圖像
    log_to_file(log_file, f"開始預測 {len(test_dataset)} 張測試圖像...")
    results = []
    
    for idx, (img, class_idx) in enumerate(tqdm(test_loader)):
        img_path = test_dataset.samples[idx][0]  # 獲取原始圖像路徑
        img_name = os.path.basename(img_path)
        class_name = test_dataset.classes[class_idx.item()]
        
        # 使用原始圖像路徑進行處理，傳入標籤參數
        process_test_image(img_path, img_name, train_features, train_file_paths, train_labels, extractor, top_k, results, log_file, class_name)
    
    display_results_summary(results, log_file)
    
    # 記錄完成時間
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(log_file, f"\n========== 預測完成 {timestamp} ==========\n")
    
    return results

def process_test_image(test_img_path, test_img_name, train_features, train_file_paths, train_labels, extractor, top_k, results, log_file=None, class_name=None):
    """處理單個測試圖像並添加結果"""
    # 直接使用extract_features從測試圖像提取特徵
    query_feature = extractor.extract_features(test_img_path)
    
    if query_feature is None:
        log_to_file(log_file, f"無法處理測試圖像: {test_img_name}")
        return
    
    # 計算與訓練集中所有特徵的余弦相似度
    similarities = []
    for idx, train_feature in enumerate(train_features):
        # 余弦相似度計算: cos(θ) = A·B / (||A||·||B||)
        sim = np.dot(query_feature, train_feature) / (np.linalg.norm(query_feature) * np.linalg.norm(train_feature))
        similarities.append((train_labels[idx], sim))
    
    # 根據相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 選擇前top_k個最相似的結果
    similar_images = similarities[:top_k]

    if not similar_images:
        log_to_file(log_file, f"無法找到相似的圖像: {test_img_name}")
        return

    # 顯示結果
    log_to_file(log_file, "\n" + "="*50)
    log_to_file(log_file, f"測試圖片: {test_img_name}" + (f" (類別: {class_name})" if class_name else ""))
    log_to_file(log_file, "-"*50)

    # 找出最相似的結果（第一個結果）
    most_similar_class, highest_similarity = similar_images[0]
    
    # 實現雙重閾值判斷邏輯
    prediction_confidence = "未知"
    prediction_reason = ""
    
    if highest_similarity > 0.8:
        # 高可信度 - 直接採用最相似圖片的類別
        prediction_confidence = "高可信度"
        prediction_reason = f"相似度 {highest_similarity:.4f} > 0.8"
    elif highest_similarity < 0.7:
        # 低可信度 - 判定為未知類別
        most_similar_class = "未知類別"
        prediction_confidence = "低可信度"
        prediction_reason = f"相似度 {highest_similarity:.4f} < 0.7"
    else:
        # 中等可信度 - 按類別分組，比較不同類別間的相似度差距
        class_best = {}
        # 將相似圖像按類別分組，每個類別只保留最高相似度
        for label, sim in similarities:
            if label not in class_best or sim > class_best[label][1]:
                class_best[label] = (label, sim)
        
        # 按相似度排序類別
        sorted_classes = sorted([(label, sim) for label, (_, sim) in class_best.items()], 
                               key=lambda x: x[1], reverse=True)
        
        # 如果只有一個類別，則採用該類別
        if len(sorted_classes) == 1:
            best_class, best_sim = sorted_classes[0]
            most_similar_class = best_class
            prediction_confidence = "中可信度-採用"
            prediction_reason = "僅有一個匹配類別"
        else:
            # 獲取最高相似度的類別 (A) 和次高相似度的類別 (B)
            best_class, best_sim = sorted_classes[0]
            second_best_class, second_best_sim = sorted_classes[1]
            similarity_gap = best_sim - second_best_sim
            
            # 若最佳與次佳類別相似度差距大於0.1，採用最佳結果
            if similarity_gap > 0.1:
                most_similar_class = best_class
                prediction_confidence = "中可信度-採用"
                prediction_reason = f"類別間相似度差距 {similarity_gap:.4f} > 0.1 (最佳:{best_class}={best_sim:.4f}, 次佳:{second_best_class}={second_best_sim:.4f})"
            else:
                most_similar_class = "未知類別"
                prediction_confidence = "中可信度-拒绝"
                prediction_reason = f"類別間相似度差距 {similarity_gap:.4f} <= 0.1 (最佳:{best_class}={best_sim:.4f}, 次佳:{second_best_class}={second_best_sim:.4f})"

    log_to_file(log_file, f"最相似類別: {most_similar_class} (相似度: {highest_similarity:.4f})")
    log_to_file(log_file, f"判斷可信度: {prediction_confidence}")
    log_to_file(log_file, f"判斷依據: {prediction_reason}")
    log_to_file(log_file, "-"*50)

    # 保存結果以便後續整理
    results.append({
        'test_image': test_img_name,
        'class_name': class_name,
        'most_similar_class': most_similar_class,
        'similarity': highest_similarity,
        'confidence': prediction_confidence,
        'reason': prediction_reason
    })

def display_results_summary(results, log_file=None):
    """顯示結果摘要"""
    log_to_file(log_file, "\n" + "="*50)
    log_to_file(log_file, "預測結果摘要:")
    log_to_file(log_file, "-"*50)
    
    # 統計成功和失敗的分類數量
    total_images = len(results)
    unknown_count = sum(1 for result in results if result['most_similar_class'] == "未知類別")
    classified_count = total_images - unknown_count
    
    # 計算百分比
    classified_percentage = (classified_count / total_images * 100) if total_images > 0 else 0
    unknown_percentage = (unknown_count / total_images * 100) if total_images > 0 else 0
    
    # 顯示每個測試圖片的結果
    for result in results:
        class_info = f" (類別: {result['class_name']})" if 'class_name' in result and result['class_name'] else ""
        log_to_file(log_file, f"{result['test_image']}{class_info} -> 類別: {result['most_similar_class']} (相似度: {result['similarity']:.4f}, {result['confidence']})")
    
    # 顯示總體統計
    log_to_file(log_file, "\n" + "="*50)
    log_to_file(log_file, "分類統計:")
    log_to_file(log_file, f"總圖片數量: {total_images}")
    log_to_file(log_file, f"成功分類數量: {classified_count} ({classified_percentage:.2f}%)")
    log_to_file(log_file, f"未能分類數量: {unknown_count} ({unknown_percentage:.2f}%)")
    log_to_file(log_file, "="*50)

# 直接執行的主程式碼
if __name__ == "__main__":
    # 要預測的參數設定
    test_dir = 'feature_db/test/'
    model_path = 'output/simclr_mobilenetv3.pth'
    top_k = 5
    db_file = 'output/train_features.db'
    
    # 確保結果目錄存在
    result_dir = 'output/result/'
    os.makedirs(result_dir, exist_ok=True)
    
    # 使用時間戳記生成唯一的檔名
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f'{result_dir}prediction_{timestamp}.txt'
    
    # 執行相似圖像預測
    print(f"\n執行相似圖像預測...\n結果將保存到: {log_file}")
    predict_and_display(test_dir, top_k, db_file, log_file)