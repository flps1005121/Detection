import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import datetime  # 導入 datetime 模組用於時間戳記

# 導入我們剛創建的特徵提取器
from feature_extractor import ImageFeatureExtractor

def log_to_file(log_file, message, print_to_console=True):
    """將訊息同時輸出到控制台和日誌檔案"""
    if print_to_console:
        print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

def predict_and_display(test_dir, top_k=5, db_file='train_features.db', log_file=None):
    """預測並顯示結果，可選擇性地記錄到檔案中"""
    # 記錄開始時間
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(log_file, f"\n========== 預測開始 {timestamp} ==========\n")
    
    # 創建特徵提取器實例
    extractor = ImageFeatureExtractor(model_path=model_path)
    
    # 使用load_features_from_database從SQLite數據庫載入特徵
    try:
        train_features, train_file_paths, train_labels = extractor.load_features_from_database(db_file)
        log_to_file(log_file, f"成功從 {db_file} 載入 {len(train_file_paths)} 個特徵向量")
    except Exception as e:
        log_to_file(log_file, f"無法載入特徵資料庫 {db_file}: {str(e)}")
        return

    if len(train_features) == 0:
        log_to_file(log_file, "無法繼續：訓練集特徵資料庫為空")
        return

    # 使用ImageFolder載入測試集圖像
    # 定義測試集的轉換函數
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
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
        
        # 使用原始圖像路徑進行處理
        process_test_image(img_path, img_name, train_features, train_file_paths, extractor, top_k, results, log_file, class_name)
    
    display_results_summary(results, log_file)
    
    # 記錄完成時間
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(log_file, f"\n========== 預測完成 {timestamp} ==========\n")
    
    return results

def process_test_image(test_img_path, test_img_name, train_features, train_file_paths, extractor, top_k, results, log_file=None, class_name=None):
    """處理單個測試圖像並添加結果"""
    # 使用更新的find_similar_images方法，不需要db_file參數，因為已經載入了features和file_paths
    similar_images = extractor.find_similar_images(
        test_img_path, 
        features=train_features, 
        file_paths=train_file_paths, 
        top_k=top_k
    )

    if not similar_images:
        log_to_file(log_file, f"無法處理測試圖像: {test_img_name}")
        return

    # 顯示結果
    log_to_file(log_file, "\n" + "="*50)
    log_to_file(log_file, f"測試圖片: {test_img_name}" + (f" (類別: {class_name})" if class_name else ""))
    log_to_file(log_file, "-"*50)

    # 找出最相似的圖片（第一個結果）
    most_similar_path, highest_similarity = similar_images[0]
    most_similar_name = os.path.basename(most_similar_path)

    log_to_file(log_file, f"最相似圖片: {most_similar_name} (相似度: {highest_similarity:.4f})")
    log_to_file(log_file, "-"*50)

    # 顯示所有相似圖片的排名
    log_to_file(log_file, "所有相似圖片排名:")
    for i, (img_path, similarity) in enumerate(similar_images):
        log_to_file(log_file, f"  {i+1}. {os.path.basename(img_path)} - 相似度: {similarity:.4f}")

    # 保存結果以便後續整理
    results.append({
        'test_image': test_img_name,
        'class_name': class_name,
        'most_similar': most_similar_name,
        'similarity': highest_similarity,
        'all_similars': [(os.path.basename(p), s) for p, s in similar_images]
    })

def display_results_summary(results, log_file=None):
    """顯示結果摘要"""
    log_to_file(log_file, "\n" + "="*50)
    log_to_file(log_file, "預測結果摘要:")
    log_to_file(log_file, "-"*50)
    for result in results:
        class_info = f" (類別: {result['class_name']})" if 'class_name' in result and result['class_name'] else ""
        log_to_file(log_file, f"{result['test_image']}{class_info} -> {result['most_similar']} (相似度: {result['similarity']:.4f})")

# 直接執行的主程式碼
if __name__ == "__main__":
    # 要預測的參數設定
    test_dir = 'feature_db_netwk/test/'
    model_path = 'output/simclr_mobilenetv3.pth'
    top_k = 5
    db_file = 'output/train_features.db'
    log_file = 'output/prediction_logs.txt'
    
    # 執行相似圖像預測
    print("\n執行相似圖像預測...")
    predict_and_display(test_dir, top_k, db_file, log_file)