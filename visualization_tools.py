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
    method='tsne',  # 'tsne' 或 'umap'
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
        reducer = umap.UMAP(n_components=2, random_state=random_state)
    else:
        raise ValueError("method 必須為 'tsne' 或 'umap'")

    reduced = reducer.fit_transform(np.nan_to_num(features))

    # 視覺化
    plt.figure(figsize=(10, 8))
    unique_labels_int = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels_int)))

    for i, label_int in enumerate(unique_labels_int):
        mask = labels == label_int
        plt.scatter(
            reduced[mask, 0], reduced[mask, 1],
            c=[colors[i]], 
            label=class_names[label_int],
            s=150, alpha=0.7, edgecolors='w'
        )

    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'{title} - {method.upper()}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
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

def visualize_GCAM_maps(
    model_path: str,
    image_path: str,
    save_path: str = 'visualizations/GCAM_map.png',
    device: torch.device = torch.device('cpu')
) -> None:
    """視覺化模型的 Grad-CAM 注意力圖，顯示模型關注的圖像區域"""
    print(f"正在為圖片生成 Grad-CAM 注意力圖：{image_path}")

    # 載入預訓練模型
    model = SimCLRNet(feature_dim=128).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"成功載入模型權重: {model_path}")
    except Exception as e:
        print(f"載入模型權重失敗: {e}")
        return

    model.eval()

    # 定義要提取的層
    target_layers = {
        'Early Layer': model.backbone.features[4],
        'Mid Layer': model.backbone.features[8],
        'Late Layer': model.backbone.features[12],
    }

    # 讀取並預處理圖像
    try:
        original_img = Image.open(image_path).convert('RGB')
        original_np = np.array(original_img)
        original_size = original_np.shape[:2]  # (H, W)
    except Exception as e:
        print(f"錯誤：無法載入圖像 {image_path}，原因：{e}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # 儲存熱圖
    heatmaps: Dict[str, np.ndarray] = {}

    # 對每個目標層生成 Grad-CAM 熱圖
    for layer_name, target_layer in target_layers.items():
        with register_hooks(target_layer) as (gradients, activations):
            # 前向傳播
            output = model(input_tensor)
            class_idx = output.argmax(dim=1).item()

            # 反向傳播
            model.zero_grad()
            target = output[0, class_idx]
            target.backward()

            # 計算 Grad-CAM
            grad = gradients[0]  # [B, C, H, W]
            activation = activations[0]  # [B, C, H, W]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            gradcam_map = F.relu((weights * activation).sum(dim=1)).squeeze().cpu().numpy()

            # 正規化並調整大小
            if gradcam_map.max() > gradcam_map.min():
                gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)
            gradcam_map = resize(gradcam_map, original_size, order=1, mode='constant', anti_aliasing=True)
            heatmaps[layer_name] = gradcam_map

    # 視覺化
    fig, axes = plt.subplots(2, len(target_layers) + 1, figsize=(16, 8))

    # 顯示原始圖像
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    # 顯示每個層的熱圖和疊加圖
    for i, (name, heatmap) in enumerate(heatmaps.items()):
        layer_idx = i + 1

        # 顯示熱圖
        axes[0, layer_idx].imshow(heatmap, cmap='jet')
        axes[0, layer_idx].set_title(f'{name} Heatmap')
        axes[0, layer_idx].axis('off')

        # 生成疊加圖
        colored_heatmap = cm.jet(heatmap)[:, :, :3]
        original_normalized = original_np.astype(float) / 255.0
        overlay = (colored_heatmap * 0.7 + original_normalized * 0.3)
        overlay = np.clip(overlay, 0, 1)

        # 顯示疊加圖
        axes[1, layer_idx].imshow(overlay)
        axes[1, layer_idx].set_title(f'{name} Overlay')
        axes[1, layer_idx].axis('off')

    # 保存結果
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Grad-CAM 熱圖已儲存於: {save_path}")
    except Exception as e:
        print(f"錯誤：無法保存圖片到 {save_path}，原因：{e}")

if __name__ == "__main__":
    # 主程式入口

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 設置文件路徑
    model_path = 'output/simclr_mobilenetv3.pth'
    #model_path = 'output/best_model.pth'
    data_dir = 'dataset/train'
    test_data_dir = 'feature_db_netwk/test/eagle'
    losses_file = 'output/training_losses.json'
    features_file = 'output/train_features.db'
    output_dir = 'output/visualizations'

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 執行視覺化任務
    print("開始視覺化分析...")

    # 繪製損失曲線
    if os.path.exists(losses_file):
        print("繪製訓練損失曲線...")
        plot_losses(losses_file, os.path.join(output_dir, 'loss_curve.png'))

    # 視覺化特徵空間 (使用已生成的特徵)
    if os.path.exists(features_file):
        print("視覺化特徵空間...")
        visualize_feature_space_from_db(
            db_file=features_file, 
            data_dir=data_dir, 
            table_name='features',
            save_path=os.path.join(output_dir, 'feature_space.png')
    )
    # # 生成注意力圖
    # if os.path.exists(model_path) and os.path.exists(test_data_dir):
    #     if test_images := [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]:
    #         print("生成模型注意力圖...")
    #         visualize_attention_maps(model_path,
    #                                 os.path.join(test_data_dir, test_images[0]),
    #                                 os.path.join(output_dir, 'attention_map.png'))

    # if os.path.exists(model_path) and os.path.exists(test_data_dir):
    #     if test_images := [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]:
    #         print("生成模型注意力圖...")
    #         visualize_GCAM_maps(model_path,
    #                                 os.path.join(test_data_dir, test_images[0]),
    #                                 os.path.join(output_dir, 'GCAM_map.png'))

    print(f"視覺化分析完成！所有結果已保存至 {output_dir} 目錄")