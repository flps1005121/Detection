import os
import random
import shutil
from pathlib import Path
import argparse

def split_dataset(train_dir, test_dir, test_ratio=0.2, copy=False):
    """
    將train目錄中的資料按照指定比例分配到test目錄
    
    參數:
    train_dir: 訓練資料的目錄路徑
    test_dir: 測試資料的目錄路徑
    test_ratio: 分配到測試集的比例 (0-1之間)
    copy: 是否複製而非移動文件
    """
    # 確保目錄存在
    train_path = Path(train_dir)
    test_path = Path(test_dir)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # 檢查訓練目錄內是否有子目錄（類別）
    all_items = list(train_path.iterdir())
    has_subdirs = any(item.is_dir() for item in all_items)
    
    if has_subdirs:
        # 如果有子目錄（類別），則保持類別平衡進行分割
        for class_dir in [item for item in all_items if item.is_dir()]:
            class_name = class_dir.name
            files = list(class_dir.glob("*"))
            random.shuffle(files)
            
            # 計算要移動到測試集的文件數量
            test_size = int(len(files) * test_ratio)
            test_files = files[:test_size]
            
            # 確保測試集中的類別目錄存在
            test_class_dir = test_path / class_name
            test_class_dir.mkdir(exist_ok=True)
            
            # 移動或複製文件
            for file_path in test_files:
                dest_path = test_class_dir / file_path.name
                if copy:
                    shutil.copy2(file_path, dest_path)
                    print(f"已複製: {file_path} 到 {dest_path}")
                else:
                    shutil.move(file_path, dest_path)
                    print(f"已移動: {file_path} 到 {dest_path}")
    else:
        # 如果沒有子目錄，直接分割文件
        files = list(train_path.glob("*"))
        random.shuffle(files)
        
        # 計算要移動到測試集的文件數量
        test_size = int(len(files) * test_ratio)
        test_files = files[:test_size]
        
        # 移動或複製文件
        for file_path in test_files:
            dest_path = test_path / file_path.name
            if copy:
                shutil.copy2(file_path, dest_path)
                print(f"已複製: {file_path} 到 {dest_path}")
            else:
                shutil.move(file_path, dest_path)
                print(f"已移動: {file_path} 到 {dest_path}")

if __name__ == "__main__":
    # 直接在這裡設定參數
    train_directory = 'feature_db/train'
    test_directory = 'feature_db/test'
    test_ratio = 0.2  # 20%的資料會被分到測試集
    copy_files = False  # 設定為True表示複製檔案，False表示移動檔案
    
    # 執行分割
    split_dataset(train_directory, test_directory, test_ratio, copy_files)
    
    print(f"資料分割完成! 測試集比例: {test_ratio}")
