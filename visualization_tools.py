import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
from PIL import Image
from torchvision import transforms
from backup_code.feature_extractor import SimCLRNet, device
import umap
import matplotlib.cm as cm
import torch.nn.functional as F
from skimage.transform import resize
from contextlib import contextmanager
from typing import Dict, List, Tuple, Generator
import sqlite3
import json
import random
from torchvision.models import mobilenet_v3_small
from contextlib import contextmanager

# 設定支援中文字體
import matplotlib
# 嘗試設置支援中文的字體
try:
    # 對於 macOS
    matplotlib.rc('font', family='Arial Unicode MS')
except:
    try:
        # 對於 Windows
        matplotlib.rc('font', family='Microsoft YaHei')
    except:
        # 對於 Linux
        try:
            matplotlib.rc('font', family='WenQuanYi Micro Hei')
        except:
            print("無法設置中文字體，視覺化中可能會出現亂碼。")


def plot_losses(losses_file, save_path='loss_curve.png'):
    # 繪製損失曲線
    if losses_file.endswith('.json'):
        with open(losses_file, 'r') as f:
            losses = json.load(f)  # 從 JSON 檔案載入損失值
    else:
        with open(losses_file, 'r') as f:
            losses = [float(line.strip()) for line in f if line.strip()]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', label='Training Loss')

    # 顯示最終損失值
    if losses:
        final_epoch = len(losses)
        final_loss = losses[-1]
        plt.annotate(f'Final Loss: \n{final_loss:.4f}',
                     xy=(final_epoch, final_loss),
                     xytext=(final_epoch, final_loss + 0.2),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                     ha='center', va='bottom')

    plt.legend()
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"損失曲線已儲存於: {save_path}")

def visualize_feature_space_from_db(
    db_file,
    data_dir,
    save_path='visualizations/feature_space.png',
    title='Feature Space Visualization',
    method='umap',  # 'tsne' 或 'umap'
    random_state=42,
    table_name='features'
):
    """
    使用 t-SNE 或 UMAP 將特徵空間降維並視覺化，從 SQLite 資料庫讀取資料。
    ...
    """
    # 從資料庫讀取資料
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # 使用一個固定的或正確的資料表名稱
    print(f"嘗試查詢資料表: {table_name}")
    try:
        cursor.execute(f"SELECT feature, label FROM {table_name}")
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError as e:
        conn.close()
        print(f"❗ SQL 查詢失敗：{e}")
        print(f"請確認資料庫 `{db_file}` 中存在名為 `{table_name}` 的資料表。")
        return
    
    if not rows:
        print("❗ 資料庫中沒有找到資料，請確認資料表名稱和內容。")
        return

    # 將資料轉換為 NumPy 陣列
    features = []
    labels = []
    # 建立標籤映射，將字串標籤轉換為整數
    class_map = {}
    class_names = []
    
    for row in rows:
        # 將 BLOB 數據轉換回 numpy 陣列
        features.append(np.frombuffer(row[0], dtype=np.float64))
        # 處理字串標籤，並將其轉換為整數
        label_str = row[1]
        if label_str not in class_map:
            class_map[label_str] = len(class_names)
            class_names.append(label_str)
        labels.append(class_map[label_str])
    
    features = np.array(features)
    labels = np.array(labels)
    
    # 降維
    if method == 'tsne':
        perplexity = max(5, min(30, features.shape[0] // 10))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    elif method == 'umap':
        # reducer = umap.UMAP(n_components=2, random_state=random_state)
        # 嘗試1：較小的 n_neighbors 來強調局部結構
        # reducer = umap.UMAP(n_neighbors=5, n_components=2, random_state=random_state)
        
        # 嘗試2：增加 min_dist 來讓群組更分散
        reducer = umap.UMAP(n_neighbors=30, min_dist=1, n_components=2, random_state=random_state)
        
        # 嘗試3：同時調整兩個參數
        # reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, n_components=2, random_state=random_state)
    else:
        raise ValueError("method 必須為 'tsne' 或 'umap'")

    reduced = reducer.fit_transform(np.nan_to_num(features))

    # 視覺化
    plt.figure(figsize=(10, 8))
    unique_labels_int = np.unique(labels)
    # colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels_int)))
    
    tab20_colors = plt.cm.get_cmap('tab10')
    for i, label_int in enumerate(unique_labels_int):
        mask = labels == label_int
        plt.scatter(
            reduced[mask, 0], reduced[mask, 1],
            # c=[colors[i]], 
            c=[tab20_colors(i % 20)],  # 使用 i % 20 確保索引值在 0 到 19 之間
            label=class_names[label_int],
            s=20, alpha=0.8, edgecolors='w' # s 是點的大小
        )

    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'{title} - {method.upper()}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if method == 'tsne':
        save_path = save_path.replace('.png', '_tsne.png')
    elif method == 'umap':
        save_path = save_path.replace('.png', '_umap.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特徵空間 ({method}) 視覺化已儲存於: {save_path}")

def plot_confusion_matrix(query_labels, preds, idx_to_class, save_path='visualizations/confusion_matrix.png'):
    """繪製混淆矩陣"""
    cm = confusion_matrix(query_labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=[idx_to_class[i] for i in range(len(idx_to_class))])
    disp.plot(cmap="Blues")
    plt.title("Query Set Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved at: {save_path}")

def visualize_attention_maps(model_path, image_path, save_path='visualizations/attention_map.png'):
    """視覺化模型的注意力圖，顯示模型關注的圖像區域"""
    # 使用已有的模型生成注意力圖
    print(f"正在為圖片生成注意力圖：{image_path}")

    # 載入預訓練模型
    model = SimCLRNet(feature_dim=128).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功載入模型權重: {model_path}")
    except Exception as e:
        print(f"載入模型權重失敗: {e}")
        return

    model.eval()

    # 使用模型的主幹網路
    backbone = model.backbone

    # 定義要提取的層 (根據MobileNetV3的結構選擇不同層級)
    target_layers = {
        'Early Layer': backbone.features[4],   # 初始特徵提取層
        'Mid Layer': backbone.features[8],   # 中間層
        'Late Layer': backbone.features[12],  # 後期層
    }

    # 保存各層特徵圖
    activation = {}

    # 註冊鉤子函數來提取特徵
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # 註冊鉤子
    hooks = []
    for name, layer in target_layers.items():
        hooks.append(layer.register_forward_hook(get_activation(name)))

    # 讀取並預處理圖像
    original_img = Image.open(image_path).convert('RGB')
    original_np = np.array(original_img)

    # 使用與訓練相同的預處理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # 進行推論以獲取特徵圖
    with torch.no_grad():
        _ = backbone.features(input_tensor)

    # 移除鉤子
    for hook in hooks:
        hook.remove()

    # 繪製視覺化
    fig, axes = plt.subplots(2, len(target_layers) + 1, figsize=(16, 8))

    # 第一個子圖顯示原始圖像
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 第一排空出一格
    axes[1, 0].axis('off')

    # 處理每一層的特徵熱圖
    for i, (name, _) in enumerate(target_layers.items()):
        layer_idx = i + 1

        # 取得特徵圖
        feature_map = activation[name]

        # 特徵圖處理：平均所有通道作為注意力圖
        attention = feature_map.mean(dim=1).squeeze().cpu().numpy()

        # 如果是單個數字，跳過這層
        if not hasattr(attention, 'shape') or len(attention.shape) < 2:
            continue

        # 正規化為 0-1 範圍
        attention_min = attention.min()
        attention_max = attention.max()
        if attention_max > attention_min:  # 避免除以零
            attention = (attention - attention_min) / (attention_max - attention_min)

        # 調整大小以匹配輸入圖像
        from skimage.transform import resize
        attention_resized = resize(attention, original_np.shape[:2],
                                  order=1, mode='constant',
                                  anti_aliasing=True)

        # 顯示熱力圖
        axes[0, layer_idx].imshow(attention, cmap='jet')
        axes[0, layer_idx].set_title(f'{name} Heatmap')
        axes[0, layer_idx].axis('off')

        # 將熱力圖映射為彩色圖
        import matplotlib.cm as cm
        colored_heatmap = cm.jet(attention_resized)[:, :, :3]

        # 疊加到原始圖像
        original_normalized = original_np.astype(float) / 255.0
        overlay = (colored_heatmap * 0.7 + original_normalized * 0.3)
        overlay = np.clip(overlay, 0, 1)

        # 顯示疊加後的圖像
        axes[1, layer_idx].imshow(overlay)
        axes[1, layer_idx].set_title(f'{name} Overlay')
        axes[1, layer_idx].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"分層特徵熱圖已儲存於: {save_path}")


@contextmanager
def register_hooks(target_layer: torch.nn.Module) -> Generator[Tuple[List[torch.Tensor], List[torch.Tensor]], None, None]:
    gradients: List[torch.Tensor] = []
    activations: List[torch.Tensor] = []
    def forward_hook(module, input, output):
        activations.append(output.detach())
    def full_backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(full_backward_hook)
    try:
        yield gradients, activations
    finally:
        forward_handle.remove()
        backward_handle.remove()

def visualize_GCAM_for_retrieval(
    model_path: str,
    query_path: str,
    positive_path: str,
    negative_path: str,
    save_path_prefix: str = 'visualizations/GCAM_map', # 改為路徑前綴
    device: torch.device = torch.device('cpu')
) -> None:
    """
    為檢索任務生成 Grad-CAM 注意力圖，並將三種對比情境分別儲存為獨立圖片。
    """
    print(f"正在為 query 圖片生成 Grad-CAM 系列圖：{query_path}")

    # 載入模型
    model = SimCLRNet(feature_dim=128).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"載入模型權重失敗: {e}")
        return
    model.eval()

    # 要提取的層
    target_layers = {
        'Early Layer': model.backbone.features[4],
        'Mid Layer': model.backbone.features[8],
        'Late Layer': model.backbone.features[12],
    }

    # 圖片讀取與預處理
    def load_image(path):
        img = Image.open(path).convert('RGB')
        original_np = np.array(img)
        size = original_np.shape[:2]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        tensor = transform(img).unsqueeze(0).to(device)
        return tensor, original_np, size

    query_tensor, query_np, query_size = load_image(query_path)
    positive_tensor, positive_np, positive_size = load_image(positive_path)
    negative_tensor, negative_np, negative_size = load_image(negative_path)
    
    # 預處理圖片，以便於疊加
    normalized_query_np = query_np.astype(float) / 255.0
    normalized_positive_np = positive_np.astype(float) / 255.0
    normalized_negative_np = negative_np.astype(float) / 255.0

    # --- 繪製圖 2a: Query vs Positive ---
    fig_qp, axes_qp = plt.subplots(1, len(target_layers) + 2, figsize=(20, 5))
    fig_qp.suptitle('Tuned Model: Query vs Positive Sample', fontsize=16)
    axes_qp[0].imshow(query_np); axes_qp[0].set_title('Query'); axes_qp[0].axis('off')
    axes_qp[1].imshow(positive_np); axes_qp[1].set_title('Positive'); axes_qp[1].axis('off')
    
    # 計算整體相似度
    cos_sim_qp_final = F.cosine_similarity(model(query_tensor), model(positive_tensor))
    
    for i, (layer_name, target_layer) in enumerate(target_layers.items()):
        with register_hooks(target_layer) as (gradients, activations):
            cos_sim = F.cosine_similarity(model(query_tensor), model(positive_tensor))
            model.zero_grad()
            cos_sim.backward(retain_graph=True) # retain_graph 以便後續計算

            grad = gradients[0]; activation = activations[0]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            gradcam_map = F.relu((weights * activation).sum(dim=1)).squeeze().cpu().numpy()
            gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)
            gradcam_map = resize(gradcam_map, query_size, order=1, mode='constant', anti_aliasing=True)
            colored = cm.jet(gradcam_map)[:, :, :3]
            overlay = np.clip(colored * 0.7 + normalized_query_np * 0.3, 0, 1)
            axes_qp[i + 2].imshow(overlay); axes_qp[i + 2].set_title(f'{layer_name}'); axes_qp[i + 2].axis('off')
    
    save_path_qp = f"{save_path_prefix}_QvP.png"
    plt.figtext(0.5, 0.01, f'Cosine Similarity: {cos_sim_qp_final.item():.4f}', ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path_qp, dpi=300, bbox_inches='tight')
    plt.close(fig_qp)
    print(f"✨ QvP 圖已儲存於: {save_path_qp}")

    # --- 繪製圖 2b: Query vs Negative ---
    fig_qn, axes_qn = plt.subplots(1, len(target_layers) + 2, figsize=(20, 5))
    fig_qn.suptitle('Tuned Model: Query vs Negative Sample', fontsize=16)
    axes_qn[0].imshow(query_np); axes_qn[0].set_title('Query'); axes_qn[0].axis('off')
    axes_qn[1].imshow(negative_np); axes_qn[1].set_title('Negative'); axes_qn[1].axis('off')
    
    cos_sim_qn_final = F.cosine_similarity(model(query_tensor), model(negative_tensor))

    for i, (layer_name, target_layer) in enumerate(target_layers.items()):
        with register_hooks(target_layer) as (gradients, activations):
            cos_sim = F.cosine_similarity(model(query_tensor), model(negative_tensor))
            model.zero_grad()
            cos_sim.backward(retain_graph=True)

            grad = gradients[0]; activation = activations[0]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            gradcam_map = F.relu((weights * activation).sum(dim=1)).squeeze().cpu().numpy()
            gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)
            gradcam_map = resize(gradcam_map, query_size, order=1, mode='constant', anti_aliasing=True)
            colored = cm.jet(gradcam_map)[:, :, :3]
            overlay = np.clip(colored * 0.7 + normalized_query_np * 0.3, 0, 1)
            axes_qn[i + 2].imshow(overlay); axes_qn[i + 2].set_title(f'Q vs N ({layer_name})'); axes_qn[i + 2].axis('off')

    save_path_qn = f"{save_path_prefix}_QvN.png"
    plt.figtext(0.5, 0.01, f'Cosine Similarity: {cos_sim_qn_final.item():.4f}', ha="center", fontsize=12, bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path_qn, dpi=300, bbox_inches='tight')
    plt.close(fig_qn)
    print(f"✨ QvN 圖已儲存於: {save_path_qn}")
    
    # --- 繪製圖 2c: Negative vs Query ---
    fig_nq, axes_nq = plt.subplots(1, len(target_layers) + 2, figsize=(20, 5))
    fig_nq.suptitle('Tuned Model: Negative vs Query Sample', fontsize=16)
    axes_nq[0].imshow(negative_np); axes_nq[0].set_title('Negative'); axes_nq[0].axis('off')
    axes_nq[1].imshow(query_np); axes_nq[1].set_title('Query'); axes_nq[1].axis('off')

    cos_sim_nq_final = F.cosine_similarity(model(negative_tensor), model(query_tensor))

    for i, (layer_name, target_layer) in enumerate(target_layers.items()):
        with register_hooks(target_layer) as (gradients, activations):
            cos_sim = F.cosine_similarity(model(negative_tensor), model(query_tensor))
            model.zero_grad()
            cos_sim.backward()

            grad = gradients[0]; activation = activations[0]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            gradcam_map = F.relu((weights * activation).sum(dim=1)).squeeze().cpu().numpy()
            gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)
            gradcam_map = resize(gradcam_map, negative_size, order=1, mode='constant', anti_aliasing=True)
            colored = cm.jet(gradcam_map)[:, :, :3]
            overlay = np.clip(colored * 0.7 + normalized_negative_np * 0.3, 0, 1)
            axes_nq[i + 2].imshow(overlay); axes_nq[i + 2].set_title(f'N vs Q ({layer_name})'); axes_nq[i + 2].axis('off')

    save_path_nq = f"{save_path_prefix}_NvQ.png"
    plt.figtext(0.5, 0.01, f'Cosine Similarity: {cos_sim_nq_final.item():.4f}', ha="center", fontsize=12, bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path_nq, dpi=300, bbox_inches='tight')
    plt.close(fig_nq)
    print(f"✨ NvQ 圖已儲存於: {save_path_nq}")

@contextmanager
def register_hooks(target_layer):
    """用於註冊前向和後向鉤子的上下文管理器"""
    gradients = []
    activations = []

    def save_gradients(module, grad_input, grad_output):
        # 這裡只取 grad_output，它包含了我們需要的梯度
        gradients.append(grad_output[0])

    def save_activations(module, input, output):
        activations.append(output)

    hook_grad = target_layer.register_backward_hook(save_gradients)
    hook_act = target_layer.register_forward_hook(save_activations)
    
    try:
        yield gradients, activations
    finally:
        hook_grad.remove()
        hook_act.remove()

def visualize_single_model_gcam(
    model: torch.nn.Module,
    image_path: str,
    save_path: str = 'visualizations/single_model_gcam.png',
    device: torch.device = torch.device('cpu')
) -> None:
    """
    為單一圖片、單一模型生成 Grad-CAM 注意力圖。
    """
    print(f"正在為圖片 {image_path} 生成單一模型的 GCAM...")

    model.to(device)
    model.to(torch.float32)
    
    # 這裡的邏輯與之前的保持一致
    if not hasattr(model, 'backbone') and hasattr(model, 'features'):
        # 處理標準 torchvision 模型的情況
        target_layers = {
            'Early Layer': model.features[4],
            'Mid Layer': model.features[8],
            'Late Layer': model.features[12],
        }
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
        # 處理 SimCLRNet 類似模型的情況
        target_layers = {
            'Early Layer': model.backbone.features[4],
            'Mid Layer': model.backbone.features[8],
            'Late Layer': model.backbone.features[12],
        }
    else:
        print("❌ 錯誤: 模型結構與預期不符，無法找到 features 或 backbone.features。")
        return
        
    # 圖片預處理... (省略，保持不變)
    image = Image.open(image_path).convert('RGB')
    original_np = np.array(image)
    size = original_np.shape[:2]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 視覺化... (省略，保持不變)
    fig, axes = plt.subplots(1, len(target_layers) + 1, figsize=(20, 5))
    normalized_original_np = original_np.astype(float) / 255.0
    axes[0].imshow(original_np); axes[0].set_title('Original Image'); axes[0].axis('off')
    
    for i, (layer_name, target_layer) in enumerate(target_layers.items()):
        with register_hooks(target_layer) as (gradients, activations):
            model.eval()
            with torch.enable_grad():
                output = model(image_tensor)
                
                if output.dim() == 1:
                    pred_class = output.argmax().item()
                else:
                    pred_class = output[0].argmax().item()

                model.zero_grad()
                output[0, pred_class].backward()

            grad = gradients[0].detach()
            activation = activations[0].detach()
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activation).sum(dim=1)).squeeze()
        
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        
        cam = resize(cam, size, order=3, mode='reflect', anti_aliasing=True)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        colored_heatmap = cm.jet(cam)[:, :, :3]
        overlay = np.clip(colored_heatmap * 0.7 + normalized_original_np * 0.3, 0, 1)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f'{layer_name}')
        axes[i + 1].axis('off')
    
    # 保存
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ GCAM 已成功保存至: {save_path}")
    except Exception as e:
        print(f"❌ 錯誤：無法保存圖片到 {save_path}，原因：{e}")

def auto_visualize_GCAM(model_path, test_data_dir, output_dir, query_img=None):
    """自動幫 query 圖片找 positive 圖片，然後呼叫 visualize_GCAM_for_retrieval"""

    # 先抓所有圖片
    test_images = [f for f in os.listdir(test_data_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    if not test_images:
        print("❌ 找不到測試圖片！")
        return

    # 如果沒指定 query，就用第一張
    if query_img is None:
        query_img = test_images[0]

    query_path = os.path.join(test_data_dir, query_img)
    
    positive_data_dir = 'feature_db/train/eagle' 
    negative_data_dir = 'feature_db/train/vending' 
    
    positive_images = [f for f in os.listdir(positive_data_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    negative_images = [f for f in os.listdir(negative_data_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    # 選一張不同的圖當 positive
    positive_img = random.choice(positive_images)
    negative_image = random.choice(negative_images)

    # possible_positives = [img for img in test_images if img != query_img]
    # if possible_positives:
    #     positive_img = random.choice(possible_positives)
    # else:
    #     # 如果只有一張圖，就 fallback 成 query 自己
    #     positive_img = query_img

    # positive_path = os.path.join(test_data_dir, positive_img)
    
    positive_path = os.path.join(positive_data_dir, positive_img)
    negative_path = os.path.join(negative_data_dir, negative_image)

    save_path = os.path.join(output_dir, f'GCAM_{os.path.splitext(query_img)[0]}_{os.path.splitext(positive_img)[0]}_{os.path.splitext(negative_image)[0]}.png')

    print(f"Query 圖片: {query_path}")
    print(f"Positive 圖片: {positive_path}")
    print(f"Negative 圖片: {negative_path}")
    print(f"輸出路徑: {save_path}")

    # 呼叫原本的函式
    visualize_GCAM_for_retrieval(
        model_path=model_path,
        query_path=query_path,
        positive_path=positive_path,
        negative_path=negative_path,
        save_path=save_path
    )

def auto_visualize_clear_GCAM(model_path, test_data_dir, output_dir, query_img=None):
    """為未微調的乾淨模型自動生成單一圖片的 GCAM"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根據錯誤訊息，創建一個可以載入權重的 MobileNetV3 模型
    # 您的權重文件似乎是來自一個與標準 torchvision 模型鍵名不同的版本
    # 因此，我們需要手動載入並處理鍵名
    model = mobilenet_v3_small(pretrained=False)

    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # 這裡加入修正鍵名的邏輯
        new_state_dict = {}
        for k, v in state_dict.items():
            # 這是針對原始權重文件，它沒有 'backbone' 或 'features' 前綴
            # 判斷是否為 MobileNetV3 的特徵層權重
            # 一個更魯棒的方法是直接判斷權重鍵名是否在模型中
            if 'features' in k or 'conv' in k:
                new_state_dict[k] = v
            # 原始程式碼中的 SimCLRNet 結構可能不同，我們需要檢查
            # if k.startswith('backbone.'): # 這是針對訓練過的 SimCLR 模型
            #     new_state_dict[k] = v
            # else: # 這是針對乾淨的 MobileNetV3
            #     new_state_dict[f'backbone.{k}'] = v

        # 載入模型，並設定 strict=False 來忽略缺少的 keys
        # 這樣就不會因為分類器等層的權重缺失而報錯
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ 成功載入模型權重: {model_path}")
    except Exception as e:
        print(f"❌ 載入模型權重失敗: {e}")
        return

    # 選擇圖片進行視覺化... (省略，保持不變)
    test_images = [f for f in os.listdir(test_data_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not test_images:
        print("❌ 找不到測試圖片！")
        return
    
    single_image_path = os.path.join(test_data_dir, test_images[0])
    save_path = os.path.join(output_dir, f'GCAM_clear_{os.path.splitext(test_images[0])[0]}.png')

    print(f"視覺化圖片: {single_image_path}")
    print(f"輸出路徑: {save_path}")

    visualize_single_model_gcam(
        model=model,
        image_path=single_image_path,
        save_path=save_path,
        device=device
    )

def visualize_comparison_gcam(
    model: torch.nn.Module,
    model_type: str, # 'simclr' 或 'classifier'
    query_path: str,
    comparison_path: str,
    save_path: str,
    comparison_label: str, # 'Positive' or 'Negative'
    device: torch.device = torch.device('cpu')
) -> None:
    """
    通用的 Grad-CAM 對比視覺化函式。
    能處理 SimCLRNet 和標準的 torchvision 分類模型。
    """
    print(f"正在生成對比圖 ({comparison_label}): {os.path.basename(query_path)} vs {os.path.basename(comparison_path)}")

    model.to(device).eval()

    # 根據模型類型決定目標層和特徵提取器
    if model_type == 'simclr':
        target_layers = {
            'Early': model.backbone.features[4],
            'Mid': model.backbone.features[8],
            'Late': model.backbone.features[12],
        }
        feature_extractor = model
    elif model_type == 'classifier':
        target_layers = {
            'Early': model.features[4],
            'Mid': model.features[8],
            'Late': model.features[12],
        }
        # 對於分類模型，我們取分類頭之前的全局平均池化層的輸出作為特徵
        feature_extractor = torch.nn.Sequential(
            model.features,
            model.avgpool
        )
    else:
        raise ValueError("model_type 必須是 'simclr' 或 'classifier'")

    # 圖片讀取與預處理
    def load_image(path):
        img = Image.open(path).convert('RGB')
        original_np = np.array(img)
        size = original_np.shape[:2]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        tensor = transform(img).unsqueeze(0).to(device)
        return tensor, original_np, size

    query_tensor, query_np, query_size = load_image(query_path)
    comparison_tensor, comparison_np, _ = load_image(comparison_path)
    
    normalized_query_np = query_np.astype(float) / 255.0

    # --- 繪製 ---
    fig, axes = plt.subplots(1, len(target_layers) + 2, figsize=(20, 5))
    fig.suptitle(f'{model_type.upper()} Model: Query vs {comparison_label} Sample', fontsize=16)
    axes[0].imshow(query_np); axes[0].set_title('Query'); axes[0].axis('off')
    axes[1].imshow(comparison_np); axes[1].set_title(comparison_label); axes[1].axis('off')

    # 計算特徵和相似度
    query_feat = feature_extractor(query_tensor).flatten(1)
    comparison_feat = feature_extractor(comparison_tensor).flatten(1)
    cos_sim_final = F.cosine_similarity(query_feat, comparison_feat)
    
    for i, (layer_name, target_layer) in enumerate(target_layers.items()):
        with register_hooks(target_layer) as (gradients, activations):
            # 重新計算特徵和相似度以觸發 backward
            q_feat = feature_extractor(query_tensor).flatten(1)
            c_feat = feature_extractor(comparison_tensor).flatten(1)
            cos_sim = F.cosine_similarity(q_feat, c_feat)
            
            model.zero_grad()
            cos_sim.backward()

            grad = gradients[0].detach()
            activation = activations[0].detach()
            weights = grad.mean(dim=(2, 3), keepdim=True)
            gradcam_map = F.relu((weights * activation).sum(dim=1)).squeeze().cpu().numpy()
            gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)
            gradcam_map = resize(gradcam_map, query_size, order=1, mode='constant', anti_aliasing=True)
            
            colored = cm.jet(gradcam_map)[:, :, :3]
            overlay = np.clip(colored * 0.7 + normalized_query_np * 0.3, 0, 1)
            axes[i + 2].imshow(overlay); axes[i + 2].set_title(f'{layer_name}'); axes[i + 2].axis('off')

    color = "orange" if comparison_label == "Positive" else "lightblue"
    plt.figtext(0.5, 0.01, f'Cosine Similarity: {cos_sim_final.item():.4f}', ha="center", fontsize=12, bbox={"facecolor": color, "alpha":0.5, "pad":5})
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✨ 對比圖已儲存於: {save_path}")

if __name__ == "__main__":
    # 主程式入口
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 設置文件路徑
    clear_model_path  = 'output/mobilenet_v3_small_place365.pth'
    tuned_model_path  = 'output/best_model.pth'
    data_dir = 'dataset/train'
    test_data_dir = 'feature_db/train/eagle' 
    output_dir = 'output/visualizations'

    losses_file = 'output/training_losses.json'
    clear_losses_file = 'output/training_losses_clear.json'
    features_file = 'output/train_features.db'
    clear_features_file = 'output/train_feature_clear.db'

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 執行視覺化任務
    print("開始視覺化分析...")

    # # 繪製損失曲線
    # if os.path.exists(losses_file):
    #     print("繪製訓練損失曲線...")
    #     plot_losses(losses_file, os.path.join(output_dir, 'mobilenetv3_loss_curve.png'))

    # # 視覺化特徵空間 (使用已生成的特徵)
    # if os.path.exists(features_file):
    #     print("視覺化特徵空間...")
    #     visualize_feature_space_from_db(
    #         db_file=features_file, 
    #         data_dir=data_dir, 
    #         table_name='features',
    #         save_path=os.path.join(output_dir, 'feature_space.png')
    # )
    # if os.path.exists(features_file):
    #     print("視覺化乾淨的MBV3特徵空間...")
    #     visualize_feature_space_from_db(
    #         db_file=clear_features_file, 
    #         data_dir=data_dir, 
    #         table_name='features',
    #         save_path=os.path.join(output_dir, 'feature_space_clear.png')
    # )
    # 生成注意力圖
    # if os.path.exists(model_path) and os.path.exists(test_data_dir):
    #     if test_images := [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]:
    #         print("生成模型注意力圖...")
    #         visualize_attention_maps(model_path,
    #                                 os.path.join(test_data_dir, test_images[0]),
    #                                 os.path.join(output_dir, 'attention_map.png'))

    # if os.path.exists(model_path) and os.path.exists(test_data_dir):
    #     if test_images := [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]:
    #         print("生成模型注意力圖...")
    #         print(test_images[0] + " vs " + test_images[-1])
    #         auto_visualize_GCAM(
    #             model_path="output/best_model.pth",
    #             test_data_dir="feature_db/train/eagle",
    #             output_dir="output/visualizations"
    #         )
    # if os.path.exists(clear_model_path) and os.path.exists(test_data_dir):
    #     if [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]:
    #         print("生成clear模型注意力圖...")
    #         auto_visualize_clear_GCAM(
    #             model_path=clear_model_path,
    #             test_data_dir=test_data_dir,
    #             output_dir=output_dir
    #         )
    # else:
    #     if not os.path.exists(clear_model_path):
    #         print(f"❌ 錯誤: 找不到模型權重文件 {clear_model_path}")
    #     if not os.path.exists(test_data_dir):
    #         print(f"❌ 錯誤: 找不到測試資料夾 {test_data_dir}")

    # print(f"視覺化分析完成！所有結果已保存至 {output_dir} 目錄")

    # 圖片路徑
    query_dir = 'feature_db/train/eagle'
    positive_dir = 'feature_db/train/eagle'
    negative_dir = 'feature_db/train/vending'
     # --- 隨機選取圖片 ---
    query_img_name = random.choice([f for f in os.listdir(query_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    positive_img_name = random.choice([f for f in os.listdir(positive_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f != query_img_name])
    negative_img_name = random.choice([f for f in os.listdir(negative_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    query_path = os.path.join(query_dir, query_img_name)
    positive_path = os.path.join(positive_dir, positive_img_name)
    negative_path = os.path.join(negative_dir, negative_img_name)
    
    print("=== 開始生成視覺化分析圖 ===")

    # --- 任務一：為『未微調模型』生成對比圖 ---
    print("\n[任務 1/2] 正在為『未微調模型』生成對比圖...")
    if os.path.exists(clear_model_path):
        untuned_model = mobilenet_v3_small(pretrained=False) #
        # 修正分類器以匹配 Places365 權重
        num_features = untuned_model.classifier[3].in_features
        untuned_model.classifier[3] = torch.nn.Linear(num_features, 365)
        untuned_model.load_state_dict(torch.load(clear_model_path, map_location=device)) #
        
        # 1a: 未微調模型 vs 正樣本
        visualize_comparison_gcam(
            model=untuned_model, model_type='classifier',
            query_path=query_path, comparison_path=positive_path,
            save_path=os.path.join(output_dir, 'G-CAM_01a_untuned_QvP.png'),
            comparison_label='Positive', device=device
        )
        # 1b: 未微調模型 vs 負樣本
        visualize_comparison_gcam(
            model=untuned_model, model_type='classifier',
            query_path=query_path, comparison_path=negative_path,
            save_path=os.path.join(output_dir, 'G-CAM_01b_untuned_QvN.png'),
            comparison_label='Negative', device=device
        )
    else:
        print(f"❌ 找不到未微調模型: {clear_model_path}")

    # --- 任務二：為『已微調模型』生成對比圖 ---
    print("\n[任務 2/2] 正在為『已微調模型』生成對比圖...")
    if os.path.exists(tuned_model_path):
        tuned_model = SimCLRNet(feature_dim=128)
        tuned_model.load_state_dict(torch.load(tuned_model_path, map_location=device)) #

        # 2a: 已微調模型 vs 正樣本
        visualize_comparison_gcam(
            model=tuned_model, model_type='simclr',
            query_path=query_path, comparison_path=positive_path,
            save_path=os.path.join(output_dir, 'G-CAM_02a_tuned_QvP.png'),
            comparison_label='Positive', device=device
        )
        # 2b: 已微調模型 vs 負樣本 (Query 視角)
        visualize_comparison_gcam(
            model=tuned_model, model_type='simclr',
            query_path=query_path, comparison_path=negative_path,
            save_path=os.path.join(output_dir, 'G-CAM_02b_tuned_QvN.png'),
            comparison_label='Negative', device=device
        )
        # 2c: 已微調模型 vs 負樣本 (Negative 視角)
        visualize_comparison_gcam(
            model=tuned_model, model_type='simclr',
            query_path=negative_path, comparison_path=query_path,
            save_path=os.path.join(output_dir, 'G-CAM_02c_tuned_NvQ.png'),
            comparison_label='Negative', device=device
        )
    else:
        print(f"❌ 找不到已微調模型: {tuned_model_path}")

    print(f"\n✅ 視覺化分析完成！所有結果已保存至 {output_dir} 目錄")