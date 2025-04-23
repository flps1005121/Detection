import os
import torch
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 導入我們剛創建的特徵提取器
from feature_extractor import ImageFeatureExtractor, SimCLRNet, infer_transform, log_print, device, test_dir

def predict_and_display(extractor, test_dir, train_dir, top_k=5, force_refresh=False):
    """只預測並顯示結果，不進行保存"""
    # 提取訓練集特徵
    train_features, train_file_paths = extractor.create_features_database(
        train_dir, cache_file='train_features.npz', force_refresh=force_refresh
    )

    if len(train_features) == 0:
        log_print("無法繼續：訓練集特徵資料庫為空")
        return

    # 獲取測試集圖像
    test_images = [f for f in os.listdir(test_dir)
                   if os.path.isfile(os.path.join(test_dir, f)) and
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))]

    if not test_images:
        log_print("警告：在測試目錄 {test_dir} 中未找到任何圖像")
        return []

    log_print(f"開始預測 {len(test_images)} 張測試圖像...")

    results = []

    for test_img in tqdm(test_images):
        test_img_path = os.path.join(test_dir, test_img)
        similar_images = extractor.find_similar_images(test_img_path, train_features, train_file_paths, top_k)

        if not similar_images:
            log_print(f"無法處理測試圖像: {test_img}")
            continue

        # 顯示結果
        log_print("\n" + "="*50)
        log_print(f"測試圖片: {test_img}")
        log_print("-"*50)

        # 找出最相似的圖片（第一個結果）
        most_similar_path, highest_similarity = similar_images[0]
        most_similar_name = os.path.basename(most_similar_path)

        log_print(f"最相似圖片: {most_similar_name} (相似度: {highest_similarity:.4f})")
        log_print("-"*50)

        # 顯示所有相似圖片的排名
        print("所有相似圖片排名:")
        for i, (img_path, similarity) in enumerate(similar_images):
            log_print(f"  {i+1}. {os.path.basename(img_path)} - 相似度: {similarity:.4f}")

        # 保存結果以便後續整理
        results.append({
            'test_image': test_img,
            'most_similar': most_similar_name,
            'similarity': highest_similarity,
            'all_similars': [(os.path.basename(p), s) for p, s in similar_images]
        })

    # 顯示所有結果的摘要
    log_print("\n" + "="*50)
    log_print("預測結果摘要:")
    log_print("-"*50)
    for result in results:
        log_print(f"{result['test_image']} -> {result['most_similar']} (相似度: {result['similarity']:.4f})")

    return results

def evaluate_few_shot(few_shot_k=3, model=None, model_path='self_supervised_mobilenetv3.pth'):
    """進行少樣本原型分類"""
    if model is None:
        extractor = ImageFeatureExtractor(model_path=model_path)
        model = extractor.model
    model.eval()

    infer_dataset = ImageFolder(root=train_dir, transform=infer_transform)
    class_to_idx = infer_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    log_print("訓練集類別索引:", class_to_idx)

    # 收集特徵嵌入
    class_to_embeddings = defaultdict(list)
    for x, y in infer_dataset:
        emb = model(x.unsqueeze(0).to(device)).squeeze(0).detach().cpu()
        class_to_embeddings[y].append(emb)

    # 分割為支持集和查詢集
    support_embeddings, support_labels = [], []
    query_embeddings, query_labels = [], []

    for class_id, embeddings in class_to_embeddings.items():
        log_print(f"類別 {idx_to_class[class_id]} 有 {len(embeddings)} 個樣本")
        random.shuffle(embeddings)
        support_embs = embeddings[:few_shot_k]
        query_embs = embeddings[few_shot_k:]
        support_embeddings.extend(support_embs)
        support_labels.extend([class_id] * len(support_embs))
        query_embeddings.extend(query_embs)
        query_labels.extend([class_id] * len(query_embs))

    support_embeddings = torch.stack(support_embeddings)
    support_labels = np.array(support_labels)
    log_print(f"支持集大小: {len(support_labels)} 個樣本")

    if not query_embeddings:
        log_print("警告: 查詢集為空")
        return None, None, None, None

    query_embeddings = torch.stack(query_embeddings)
    query_labels = np.array(query_labels)
    log_print(f"查詢集大小: {len(query_labels)} 個樣本")

    # 計算原型
    support_prototypes = torch.stack([
        support_embeddings[support_labels == c].mean(dim=0)
        for c in range(len(class_to_idx))
    ])

    # 預測
    preds = []
    for i, q in enumerate(query_embeddings):
        dists = torch.norm(support_prototypes - q, dim=1)
        pred_class = torch.argmin(dists).item()
        preds.append(pred_class)
        log_print(f"查詢 {i}: 真實類別 = {idx_to_class[query_labels[i]]}, 預測類別 = {idx_to_class[pred_class]}")

    acc = np.mean(np.array(preds) == query_labels)
    log_print(f"少樣本原型分類準確率: {acc:.2f}")

    return support_prototypes, query_embeddings, query_labels, preds

def evaluate_test(support_prototypes, model=None, model_path='self_supervised_mobilenetv3.pth'):
    """評估測試集"""
    if model is None:
        extractor = ImageFeatureExtractor(model_path=model_path)
        model = extractor.model
    model.eval()

    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        log_print(f"測試目錄 '{test_dir}' 不存在或為空")
        return None

    test_dataset = ImageFolder(root=test_dir, transform=infer_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    log_print("測試集類別索引:", test_dataset.class_to_idx)

    correct, total = 0, 0
    with torch.no_grad():
        for x, label in test_loader:
            emb = model(x.to(device)).squeeze(0).cpu()
            dists = torch.norm(support_prototypes - emb, dim=1)
            pred_class = torch.argmin(dists).item()
            log_print(f"真實類別: {test_dataset.classes[label.item()]}, 預測類別: {test_dataset.classes[pred_class]}")
            correct += (pred_class == label.item())
            total += 1
    test_acc = correct / total
    log_print(f"測試集準確率: {test_acc:.2f}")
    return test_acc

def auto_experiment(model=None, model_path='self_supervised_mobilenetv3.pth'):
    """自動實驗不同 few_shot_k 值"""
    if model is None:
        extractor = ImageFeatureExtractor(model_path=model_path)
        model = extractor.model
    model.eval()

    infer_dataset = ImageFolder(root=train_dir, transform=infer_transform)
    class_to_idx = infer_dataset.class_to_idx
    class_to_embeddings = defaultdict(list)
    for x, y in infer_dataset:
        emb = model(x.unsqueeze(0).to(device)).squeeze(0).detach().cpu()
        class_to_embeddings[y].append(emb)

    max_k = min(len(v) for v in class_to_embeddings.values())
    results = []
    test_loader = DataLoader(ImageFolder(root=test_dir, transform=infer_transform), batch_size=1, shuffle=False)

    log_print(f"\n[自動實驗] 測試 few_shot_k = 1 到 {max_k}")
    for k in range(1, max_k + 1):
        log_print(f"\n[自動實驗] --- 測試 few_shot_k = {k} ---")
        support_embeddings, support_labels = [], []
        query_embeddings, query_labels = [], []

        for class_id, embeddings in class_to_embeddings.items():
            random.shuffle(embeddings)
            support_embs = embeddings[:k]
            query_embs = embeddings[k:]
            support_embeddings.extend(support_embs)
            support_labels.extend([class_id] * len(support_embs))
            query_embeddings.extend(query_embs)
            query_labels.extend([class_id] * len(query_embs))

        support_embeddings = torch.stack(support_embeddings)
        support_labels = np.array(support_labels)
        query_embeddings = torch.stack(query_embeddings)
        query_labels = np.array(query_labels)

        support_prototypes = torch.stack([
            support_embeddings[support_labels == c].mean(dim=0)
            for c in range(len(class_to_idx))
        ])

        # 查詢集分類
        preds = []
        for q in query_embeddings:
            dists = torch.norm(support_prototypes - q, dim=1)
            pred_class = torch.argmin(dists).item()
            preds.append(pred_class)
        train_acc = np.mean(np.array(preds) == query_labels)

        # 測試集評估
        correct, total = 0, 0
        with torch.no_grad():
            for x, label in test_loader:
                emb = model(x.to(device)).squeeze(0).cpu()
                dists = torch.norm(support_prototypes - emb, dim=1)
                pred_class = torch.argmin(dists).item()
                correct += (pred_class == label.item())
                total += 1
        test_acc = correct / total

        log_print(f"→ 訓練集準確率 = {train_acc:.2f}, 測試集準確率 = {test_acc:.2f}")
        results.append((k, train_acc, test_acc))

    with open("results_log.txt", "w", encoding='utf-8') as f:
        f.write("few_shot_k, 訓練集準確率, 測試集準確率\n")
        for k, train_acc, test_acc in results:
            f.write(f"few_shot_k={k}, 訓練集準確率={train_acc:.2f}, 測試集準確率={test_acc:.2f}\n")

    return results

# 直接執行的主程式碼，不需要命令列參數
if __name__ == "__main__":
    # 要預測的參數設定 (可以直接修改這裡的值)
    test_dir = 'dataset/test'  # 測試圖像目錄
    train_dir = 'dataset/train'  # 訓練圖像目錄
    model_path = 'self_supervised_mobilenetv3.pth'  # 模型路徑
    top_k = 5  # 顯示最相似的K張圖像
    force_refresh = False  # 是否強制重新生成特徵資料庫
    few_shot_k = 3  # 少樣本學習的樣本數

    # 檢查目錄是否存在
    if not os.path.exists(test_dir):
        print(f"錯誤：測試目錄 {test_dir} 不存在")
    elif not os.path.exists(train_dir):
        print(f"錯誤：訓練目錄 {train_dir} 不存在")
    else:
        # 創建特徵提取器
        extractor = ImageFeatureExtractor(model_path=model_path)

        # 進行相似圖像預測並顯示結果
        log_print("\n執行相似圖像預測...")
        predict_and_display(extractor, test_dir, train_dir, top_k, force_refresh)

        # 進行少樣本原型分類
        log_print("\n執行少樣本原型分類...")
        support_prototypes, query_embeddings, query_labels, preds = evaluate_few_shot(few_shot_k, model=extractor.model)