import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import datetime
import sqlite3

# 導入特徵提取器
from backup_code.prototype_db import ImageFeatureExtractor

def log_to_file(log_file, message, print_to_console=True):
    """將訊息同時輸出到控制台和日誌檔案"""
    if print_to_console:
        print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

def load_prototypes_from_database(db_file):
    """從SQLite數據庫讀取原型特徵數據"""
    if not os.path.exists(db_file):
        print(f"錯誤：特徵數據庫 {db_file} 不存在")
        return {}
    
    try:
        # 連接到SQLite數據庫
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # 檢查原型表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prototypes'")
        if not cursor.fetchone():
            print("警告：數據庫中沒有原型表")
            conn.close()
            return {}
            
        # 讀取所有原型
        cursor.execute("SELECT label, feature, sample_count FROM prototypes")
        rows = cursor.fetchall()
        
        prototypes = {}
        for label, feature_bytes, sample_count in rows:
            # 將二進制數據轉換回numpy數組
            feature = np.frombuffer(feature_bytes, dtype=np.float32)
            prototypes[label] = {
                'feature': feature,
                'sample_count': sample_count
            }
            
        conn.close()
        print(f"已從 {db_file} 中載入 {len(prototypes)} 個類別的原型特徵")
        return prototypes
    except Exception as e:
        print(f"從數據庫讀取原型時發生錯誤: {e}")
        return {}

def cosine_similarity(vec1, vec2):
    """計算兩個向量之間的餘弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def predict_and_display(test_dir, db_file='train_features.db', log_file=None, selected_label=None, 
                        threshold=0.75):
    """使用原型特徵預測並顯示結果"""
    # 創建特徵提取器實例
    extractor = ImageFeatureExtractor(model_path=model_path)
    
    # 載入原型特徵
    prototypes = load_prototypes_from_database(db_file)
    if not prototypes:
        log_to_file(log_file, "錯誤：無法載入原型特徵，無法進行比對")
        return
    else:
        log_to_file(log_file, f"已載入 {len(prototypes)} 個類別的原型特徵")
    
    # 如果指定了特定標籤，只保留該標籤的原型
    if selected_label and selected_label in prototypes:
        prototypes = {selected_label: prototypes[selected_label]}
        log_to_file(log_file, f"只使用標籤 '{selected_label}' 的原型進行比對")

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
        
        # 使用原型處理每張圖像
        process_test_image_with_prototype(
            img_path, img_name, prototypes, extractor, results, 
            threshold, log_file, class_name
        )
    
    # 顯示結果摘要
    display_results_summary(results, log_file)
    
    return results

def process_test_image_with_prototype(test_img_path, test_img_name, prototypes, extractor, 
                                      results, threshold, log_file=None, class_name=None):
    """使用原型處理單個測試圖像"""
    # 提取特徵
    query_feature = extractor.extract_features(test_img_path)
    
    if query_feature is None:
        return
    
    # 計算與所有原型的相似度
    prototype_similarities = []
    for label, prototype_data in prototypes.items():
        prototype_feature = prototype_data['feature']
        # 餘弦相似度計算
        sim = cosine_similarity(query_feature, prototype_feature)
        prototype_similarities.append((label, sim, prototype_data['sample_count']))
    
    # 根據相似度排序
    prototype_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 只處理最相似的結果
    if not prototype_similarities:
        return

    # 找出最相似的原型
    most_similar_label, highest_similarity, sample_count = prototype_similarities[0]

    # 閾值判斷機制
    if highest_similarity >= threshold:
        decision = "接受"
        final_label = most_similar_label
    else:
        decision = "拒絕"
        final_label = "??"
    
    # 保存結果以便摘要顯示
    results.append({
        'test_image': test_img_name,
        'class_name': class_name,
        'most_similar_label': most_similar_label,
        'similarity': highest_similarity,
        'sample_count': sample_count,
        'decision': decision,
        'final_label': final_label
    })

def display_results_summary(results, log_file=None):
    """顯示結果摘要，只包含原型比對結果"""
    # 定義每個欄位的寬度
    col_widths = {
        'Image': 25,
        'Label': 10,
        'Similar': 7,
        '決策': 2
    }
    
    # 格式化表頭
    header = f"{'Image':<{col_widths['Image']}} | {'Label':<{col_widths['Label']}} | {'Similar':<{col_widths['Similar']}} | {'決策':<{col_widths['決策']}}"
    separator = "-" * len(header)
    
    log_to_file(log_file, "預測結果摘要 (只使用原型比對):")
    log_to_file(log_file, separator)
    log_to_file(log_file, header)
    log_to_file(log_file, separator)
    
    # 統計接受和拒絕的數量
    accepted = 0
    rejected = 0
    
    for result in results:
        decision_str = result['decision']
        if "接受" in decision_str:
            accepted += 1
        else:
            rejected += 1
        
        # 先格式化浮點數，再設定寬度對齊
        sim_str = f"{result['similarity']:.4f}"
        line = f"{result['test_image']:<{col_widths['Image']}} | {result['final_label']:<{col_widths['Label']}} | {sim_str:<{col_widths['Similar']}} | {decision_str:<{col_widths['決策']}}"
        log_to_file(log_file, line)
    
    # 顯示統計
    log_to_file(log_file, separator)
    log_to_file(log_file, f"總共處理: {len(results)} 張圖像")
    log_to_file(log_file, f"接受分類: {accepted} 張 ({accepted/len(results)*100:.1f}%)")
    log_to_file(log_file, f"拒絕分類: {rejected} 張 ({rejected/len(results)*100:.1f}%)")

# 直接執行的主程式碼
if __name__ == "__main__":
    # 要預測的參數設定
    test_dir = 'feature_db/test/'
    model_path = 'output/simclr_mobilenetv3.pth'
    db_file = 'output/train_features.db'
    selected_label = None  # 預設為None，表示比對所有標籤
    
    # 相似度閾值設定
    threshold = 0.7  # 閾值：高於此值接受，低於此值拒絕
    
    # 確保結果目錄
    result_dir = 'output/result/'
    os.makedirs(result_dir, exist_ok=True)
    
    # 使用時間戳記生成唯一的檔名
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label_info = f"_{selected_label}" if selected_label else ""
    log_file = f'{result_dir}prediction_prototype{label_info}_{timestamp}.txt'
    
    # 執行相似圖像預測
    print(f"\n執行相似圖像預測（僅使用原型判斷）...\n結果將保存到: {log_file}")
    
    # 執行預測
    predict_and_display(test_dir, db_file, log_file, selected_label, threshold)