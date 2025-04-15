import os
import re
import numpy as np
import pandas as pd

# 資料夾路徑，存放所有 epoch_log_*.txt 檔案
log_dir = "D:/ntou/113-2/Detection/Exp/2e-4_200"
output_file = "stability_analysis.csv"

# 檔案名稱與 lr、epochs、round 的映射表
file_mapping = {
    "epoch_log_1.txt": {"lr": "2e-4", "epochs": 200, "round": 1},
    "epoch_log_2.txt": {"lr": "2e-4", "epochs": 200, "round": 2},
    "epoch_log_3.txt": {"lr": "2e-4", "epochs": 200, "round": 3},
    "epoch_log_4.txt": {"lr": "2e-4", "epochs": 200, "round": 4},
    "epoch_log_5.txt": {"lr": "2e-4", "epochs": 200, "round": 5},
}

# 確保資料夾存在
if not os.path.exists(log_dir):
    print(f"資料夾 {log_dir} 不存在，請確認路徑！")
    exit()

# 儲存所有分析結果
results = []

# 遍歷資料夾中的所有 epoch_log_*.txt 檔案
for filename in os.listdir(log_dir):
    if filename.startswith("epoch_log_") and filename.endswith(".txt"):
        filepath = os.path.join(log_dir, filename)

        # 檢查檔案是否在映射表中
        if filename not in file_mapping:
            print(f"檔案 {filename} 不在映射表中，跳過！")
            continue

        # 提取 lr、epochs、round
        lr = file_mapping[filename]["lr"]
        epochs = file_mapping[filename]["epochs"]
        round_num = file_mapping[filename]["round"]

        # 讀取檔案並提取 Loss 值、準確率和混淆資訊
        losses = []
        few_shot_acc = None
        test_acc = None
        confusion_pairs = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取 Loss
                if "Epoch" in line and "Loss" in line:
                    loss_match = re.search(r"Loss: ([\d.]+)", line)
                    if loss_match:
                        loss = float(loss_match.group(1))
                        losses.append(loss)

                # 提取 Few-shot 準確率
                if "Few-shot prototype classification accuracy" in line:
                    acc_match = re.search(r"accuracy: ([\d.]+)", line)
                    if acc_match:
                        few_shot_acc = float(acc_match.group(1))

                # 提取 Test 準確率
                if "Test accuracy" in line:
                    acc_match = re.search(r"accuracy: ([\d.]+)", line)
                    if acc_match:
                        test_acc = float(acc_match.group(1))

                # 提取混淆資訊
                if "GT:" in line and "Pred:" in line:
                    gt_match = re.search(r"GT: ([\w_]+)", line)
                    pred_match = re.search(r"Pred: ([\w_]+)", line)
                    if gt_match and pred_match:
                        gt_class = gt_match.group(1)
                        pred_class = pred_match.group(1)
                        if gt_class != pred_class:
                            confusion_pairs.append(f"{gt_class} 誤判為 {pred_class}")

        if not losses:
            print(f"檔案 {filename} 中沒有 Loss 數據，跳過！")
            continue

        # 計算穩定性指標
        losses = np.array(losses)

        # 1. 標準差
        loss_std = np.std(losses)

        # 2. 平均相鄰 Epoch 差異
        differences = [abs(losses[i] - losses[i-1]) for i in range(1, len(losses))]
        avg_difference = np.mean(differences) if differences else 0

        # 3. 後期波動（最後 20 個 Epoch）
        late_losses = losses[-20:] if len(losses) >= 20 else losses
        late_fluctuation = max(late_losses) - min(late_losses)

        # 4. 整體波動（max - min）
        overall_fluctuation = max(losses) - min(losses)

        # 5. 最終 Loss 和最低 Loss
        final_loss = losses[-1]
        min_loss = min(losses)
        min_loss_epoch = np.argmin(losses) + 1

        # 混淆資訊
        confusion_text = ", ".join(confusion_pairs) if confusion_pairs else "無混淆"

        # 儲存結果
        results.append({
            "Learning Rate": lr,
            "Epochs": epochs,
            "Round": round_num,
            "Final Loss": final_loss,
            "Min Loss": min_loss,
            "Min Loss Epoch": min_loss_epoch,
            "Overall Fluctuation (max-min)": overall_fluctuation,
            "Standard Deviation": loss_std,
            "Avg Adjacent Epoch Difference": avg_difference,
            "Late Fluctuation (Last 20 Epochs)": late_fluctuation,
            "Few-shot Accuracy (k=3)": few_shot_acc,
            "Test Accuracy": test_acc,
            "Confusion": confusion_text
        })

# 將結果轉為 DataFrame 並排序
df = pd.DataFrame(results)
df = df.sort_values(by=["Learning Rate", "Epochs", "Round"])

# 計算每組（lr, epochs）的平均值
grouped = df.groupby(["Learning Rate", "Epochs"]).mean(numeric_only=True).reset_index()
grouped["Round"] = "Average"
grouped["Confusion"] = "-"  # 混淆資訊不計算平均值

# 合併平均值到原始數據
df = pd.concat([df, grouped], ignore_index=True)
df = df.sort_values(by=["Learning Rate", "Epochs", "Round"])

# 儲存到 CSV，使用 utf-8-sig 編碼以解決中文亂碼問題
df.to_csv(output_file, index=False, float_format="%.4f", encoding='utf-8-sig')
print(f"分析結果已儲存到 {output_file}")