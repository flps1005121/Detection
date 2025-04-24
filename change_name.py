import os
import shutil
from collections import defaultdict

def rename_images(dataset_path, subclass_method='directory'):
    """
    將數據集中的圖像重命名為 {子類別}_{數字}.{原本格式}
    
    參數:
    - dataset_path: 數據集路徑
    - subclass_method: 子類別獲取方式
    """
    # 檢查目錄是否存在
    if not os.path.isdir(dataset_path):
        print(f"錯誤：目錄 '{dataset_path}' 不存在")
        return
    
    # 用於存儲每個子類別的計數
    class_counters = defaultdict(int)
    
    # 遍歷目錄及子目錄
    for root, _, files in os.walk(dataset_path):
        # 確定當前目錄的子類別
        if subclass_method == 'directory':
            if root == dataset_path:  # 跳過頂層目錄
                continue
            subclass = os.path.basename(root)
        elif subclass_method == 'manual':
            subclass = input(f"請為目錄 '{root}' 指定子類別名稱: ")
        else:  # prefix 方法在處理每個文件時單獨處理
            subclass = None
            
        # 處理當前目錄中的文件
        for file in files:
            # 檢查是否為圖像文件
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', 'webp')):
                file_path = os.path.join(root, file)
                
                # 獲取文件擴展名，保留原始格式
                file_name, ext = os.path.splitext(file)
                
                # 如果使用前綴方法，則從文件名提取子類別
                if subclass_method == 'prefix':
                    name_parts = file_name.split('-', 1)
                    if len(name_parts) > 1:
                        subclass = name_parts[0]
                    else:
                        print(f"跳過 {file}：無法識別子類別")
                        continue
                
                # 更新計數器並生成新文件名
                class_counters[subclass] += 1
                new_filename = f"{subclass}_{class_counters[subclass]}{ext}"
                new_file_path = os.path.join(root, new_filename)
                
                # 重命名文件
                try:
                    os.rename(file_path, new_file_path)
                    print(f"重命名：{file} -> {new_filename}")
                except Exception as e:
                    print(f"重命名 {file} 時出錯：{str(e)}")

if __name__ == "__main__":
    dataset_path = 'dataset/train/'
    
    print("使用目錄名稱作為子類別（假設每個子目錄對應一個子類別）")
    
    rename_images(dataset_path, "directory")
    print("重命名完成！")
