import os
import csv  # 用來寫 CSV 檔案

# 設定資料夾路徑
folder_path = "D:/ntou/113-2/Detection/Exp/2e-4_200"

# 檢查資料夾是否存在
if not os.path.exists(folder_path):
    print(f"錯誤：資料夾 {folder_path} 不存在！請確認路徑是否正確。")
    exit()

# 準備一個列表來存所有結果
results = []

# 用來存平均值的數據
final_losses = []  # 最終 Loss
min_losses = []    # 最低 Loss
few_shot_accs = [] # Few-shot 準確率
test_accs = []     # Test 準確率

# 找到資料夾裡所有的 .txt 檔案
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # 只處理 .txt 檔案
        file_path = os.path.join(folder_path, filename)

        # 讀取檔案
        losses = []  # 存 Loss 值的列表
        few_shot_acc = None  # 存 Few-shot 準確率
        test_acc = None      # 存 Test 準確率
        try:
            # 用 UTF-8 編碼讀取檔案
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # 提取 Loss 值
                    if "Epoch" in line and "Loss" in line:
                        try:
                            loss_part = line.split("Loss: ")[1].strip()
                            loss = float(loss_part)
                            losses.append(loss)
                        except (IndexError, ValueError):
                            print(f"檔案 {filename} 的這一行格式有問題，跳過：{line.strip()}")
                            continue
                    # 提取 Few-shot 準確率
                    elif "Few-shot prototype classification accuracy" in line:
                        try:
                            few_shot_acc = float(line.split("Few-shot prototype classification accuracy: ")[1].strip())
                        except (IndexError, ValueError):
                            print(f"檔案 {filename} 的 Few-shot prototype classification accuracy 格式有問題，跳過：{line.strip()}")
                    # 提取 Test 準確率
                    elif "Test accuracy" in line:
                        try:
                            test_acc = float(line.split("Test accuracy: ")[1].strip())
                        except (IndexError, ValueError):
                            print(f"檔案 {filename} 的 Test accuracy 格式有問題，跳過：{line.strip()}")

            # 檢查是否有數據
            if not losses:
                print(f"檔案 {filename} 沒有 Loss 數據，跳過！")
                continue

            # 檢查是否有 Few-shot 和 Test 準確率
            if few_shot_acc is None or test_acc is None:
                print(f"檔案 {filename} 缺少 Few-shot 或 Test 準確率，跳過！")
                continue

            # 找出最小值和它的輪數
            min_loss = min(losses)  # 最小值
            min_epoch = losses.index(min_loss) + 1  # 輪數（從 1 開始）

            # 找出最大值
            max_loss = max(losses)  # 最大值

            # 算波動大小
            fluctuation = max_loss - min_loss

            # 取出最終 Loss（第 100 輪）
            final_loss = losses[-1]  # 最後一輪的 Loss

            # 存起來計算平均用
            final_losses.append(final_loss)
            min_losses.append(min_loss)
            few_shot_accs.append(few_shot_acc)
            test_accs.append(test_acc)

            # 把結果存起來
            results.append({
                "filename": filename,
                "final_loss": final_loss,
                "min_loss": min_loss,
                "min_epoch": min_epoch,
                "fluctuation": fluctuation,
                "few_shot_acc": few_shot_acc,
                "test_acc": test_acc
            })

        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤：{e}，跳過！")
            continue

# 檢查是否有結果
if not results:
    print("沒有任何檔案被處理，請確認資料夾裡是否有正確的 .txt 檔案！")
    exit()

# 計算平均值
avg_final_loss = sum(final_losses) / len(final_losses) if final_losses else 0
avg_min_loss = sum(min_losses) / len(min_losses) if min_losses else 0
avg_few_shot_acc = sum(few_shot_accs) / len(few_shot_accs) if few_shot_accs else 0
avg_test_acc = sum(test_accs) / len(test_accs) if test_accs else 0

# 把結果存成 CSV 檔案
output_file = "results.csv"
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    # 寫表頭
    writer.writerow([
        "檔案名稱", "最終 Loss", "最低 Loss", "最低 Loss 在第幾輪",
        "Loss 波動", "Few-shot 準確率", "Test 準確率"
    ])
    # 寫每個檔案的結果
    for result in results:
        writer.writerow([
            result['filename'],
            f"{result['final_loss']:.4f}",
            f"{result['min_loss']:.4f}",
            result['min_epoch'],
            f"{result['fluctuation']:.4f}",
            f"{result['few_shot_acc']:.4f}",
            f"{result['test_acc']:.4f}"
        ])
    # 寫平均值
    writer.writerow([])  # 空行
    writer.writerow(["平均值", f"{avg_final_loss:.4f}", f"{avg_min_loss:.4f}", "", "", f"{avg_few_shot_acc:.4f}", f"{avg_test_acc:.4f}"])

# 也印出來看看
print("檔案名稱\t\t最終 Loss\t最低 Loss(第x輪)\tLoss 波動\tFew-shot 準確率\tTest 準確率")
for result in results:
    print(f"{result['filename']:<15}\t{result['final_loss']:.4f}\t\t{result['min_loss']:.4f}(第 {result['min_epoch']} 輪)\t{result['fluctuation']:.4f}\t\t{result['few_shot_acc']:.4f}\t\t{result['test_acc']:.4f}")
print()
print(f"平均最終 Loss: {avg_final_loss:.4f}")
print(f"平均最低 Loss: {avg_min_loss:.4f}")
print(f"平均 Few-shot 準確率: {avg_few_shot_acc:.4f}")
print(f"平均 Test 準確率: {avg_test_acc:.4f}")