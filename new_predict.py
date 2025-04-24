import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 導入我們剛創建的特徵提取器
from feature_extractor import ImageFeatureExtractor

def predict_and_display(test_dir, train_dir, top_k=5, npz_file='train_features.npz'):
    """只預測並顯示結果，不進行保存"""
    # 直接加載已存在的特徵資料庫
    try:
        data = np.load(npz_file)
        train_features = data['features']
        train_file_paths = data['file_paths']
        print(f"成功從 {npz_file} 載入 {len(train_file_paths)} 個特徵向量")
    except Exception as e:
        print(f"無法載入特徵資料庫 {npz_file}: {str(e)}")
        return

    if len(train_features) == 0:
        print("無法繼續：訓練集特徵資料庫為空")
        return

    # 使用ImageFolder載入測試集圖像
    # 定義測試集的轉換函數
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"成功載入測試集: {len(test_dataset)} 張圖像, {len(test_dataset.classes)} 個類別")
    print(f"類別映射: {test_dataset.class_to_idx}")
    
    # 使用ImageFolder處理測試圖像
    print(f"開始預測 {len(test_dataset)} 張測試圖像...")
    results = []
    
    # 建立特徵提取器，但只用於查找相似圖像的比對
    extractor = ImageFeatureExtractor(model_path=model_path)
    
    for idx, (img, class_idx) in enumerate(tqdm(test_loader)):
        img_path = test_dataset.samples[idx][0]  # 獲取原始圖像路徑
        img_name = os.path.basename(img_path)
        class_name = test_dataset.classes[class_idx.item()]
        
        # 將張量轉換回PIL圖像以傳遞給find_similar_images
        process_test_image(img_path, img_name, train_features, train_file_paths, extractor, top_k, results, class_name)
    
    display_results_summary(results)
    return results

def process_test_image(test_img_path, test_img_name, train_features, train_file_paths, extractor, top_k, results, class_name=None):
    """處理單個測試圖像並添加結果"""
    similar_images = extractor.find_similar_images(test_img_path, train_features, train_file_paths, top_k)

    if not similar_images:
        print(f"無法處理測試圖像: {test_img_name}")
        return

    # 顯示結果
    print("\n" + "="*50)
    print(f"測試圖片: {test_img_name}" + (f" (類別: {class_name})" if class_name else ""))
    print("-"*50)

    # 找出最相似的圖片（第一個結果）
    most_similar_path, highest_similarity = similar_images[0]
    most_similar_name = os.path.basename(most_similar_path)

    print(f"最相似圖片: {most_similar_name} (相似度: {highest_similarity:.4f})")
    print("-"*50)

    # 顯示所有相似圖片的排名
    print("所有相似圖片排名:")
    for i, (img_path, similarity) in enumerate(similar_images):
        print(f"  {i+1}. {os.path.basename(img_path)} - 相似度: {similarity:.4f}")

    # 保存結果以便後續整理
    results.append({
        'test_image': test_img_name,
        'class_name': class_name,
        'most_similar': most_similar_name,
        'similarity': highest_similarity,
        'all_similars': [(os.path.basename(p), s) for p, s in similar_images]
    })

def display_results_summary(results):
    """顯示結果摘要"""
    print("\n" + "="*50)
    print("預測結果摘要:")
    print("-"*50)
    for result in results:
        class_info = f" (類別: {result['class_name']})" if 'class_name' in result and result['class_name'] else ""
        print(f"{result['test_image']}{class_info} -> {result['most_similar']} (相似度: {result['similarity']:.4f})")

# 直接執行的主程式碼，不需要命令列參數
if __name__ == "__main__":
    # 要預測的參數設定 (可以直接修改這裡的值)
    test_dir = 'dataset/test'  # 測試圖像目錄
    train_dir = 'dataset/train'  # 訓練圖像目錄
    model_path = 'self_supervised_mobilenetv3.pth'  # 模型路徑
    top_k = 5  # 顯示最相似的K張圖像
    npz_file = 'train_features.npz'  # 預先計算好的特徵檔案

    # 檢查目錄與檔案是否存在
    if not os.path.exists(test_dir):
        print(f"錯誤：測試目錄 {test_dir} 不存在")
    elif not os.path.exists(npz_file):
        print(f"錯誤：特徵資料庫檔案 {npz_file} 不存在")
    else:
        # 使用特徵資料庫進行預測
        print("\n執行相似圖像預測...")
        predict_and_display(test_dir, train_dir, top_k, npz_file)