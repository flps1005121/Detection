import os
import datetime
import numpy as np
import sqlite3

def log_to_file(log_file, message):
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    print(message)

# 載入資料庫中的特徵 (正確的)
def load_features_from_database(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, label, feature FROM features")
    rows = cursor.fetchall()

    file_paths = []
    labels = []
    features = []

    for file_path, label, feature_bytes in rows:
        # 將二進制數據轉換回numpy數組
        feature = np.frombuffer(feature_bytes, dtype=np.float32)

        file_paths.append(file_path)
        labels.append(label)
        features.append(feature)

    conn.close()
    return features, file_paths, labels

# 結合了old_compare.py中的兩個函數的功能
def predict_from_feature_vector(feature_vector, db_file, log_file=None, test_img_name="未知圖片", class_name=None):
    # 記錄開始時間
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(log_file, f"\n========== 特徵向量比對開始 {timestamp} ==========\n")

    # 載入資料庫特徵
    train_features, train_file_paths, train_labels = load_features_from_database(db_file)
    if len(train_features) == 0:
        log_to_file(log_file, "訓練資料特徵為空，無法進行比對")
        return []

    # 計算所有相似度
    similarities = []
    for idx, train_feature in enumerate(train_features):
        similarity = np.dot(feature_vector, train_feature) / (
            np.linalg.norm(feature_vector) * np.linalg.norm(train_feature)
        )
        similarities.append((similarity, train_labels[idx]))

    # 排序（修正排序邏輯）
    similarities.sort(key=lambda x: x[0], reverse=True)  # 注意這邊是 x[0] 而不是 x[1]

    # 選擇第一個最相似的結果
    similar_images = similarities[:1]
    highest_similarity, most_similar_class = similar_images[0]

    # 雙重門檻判定
    prediction_confidence = "未知"
    prediction_reason = ""

    if highest_similarity > 0.8:
        # 高可信度 - 直接採用最相似圖片的類別
        prediction_confidence = "高可信度"
        prediction_reason = f"相似度 {highest_similarity:.4f} > 0.8"
    elif highest_similarity < 0.7:
        # 低可信度 - 判定為未知類別
        most_similar_class = "未知類別"
        prediction_confidence = "低可信度"
        prediction_reason = f"相似度 {highest_similarity:.4f} < 0.7"
    else:
        # 中等可信度 - 按類別分組，比較不同類別間的相似度差距
        class_best = {}
        # 將相似圖像按類別分組，每個類別只保留最高相似度
        for sim, label in similarities:
            if label not in class_best or sim > class_best[label][1]:
                class_best[label] = (label, sim)

        # 按相似度排序類別
        sorted_classes = sorted([(label, sim) for label, (_, sim) in class_best.items()],
                                key=lambda x: x[1], reverse=True)

        # 如果只有一個類別，則採用該類別
        if len(sorted_classes) == 1:
            best_class, best_sim = sorted_classes[0]
            most_similar_class = best_class
            prediction_confidence = "中可信度-採用"
            prediction_reason = "僅有一個匹配類別"
        else:
            # 獲取最高相似度的類別 (A) 和次高相似度的類別 (B)
            best_class, best_sim = sorted_classes[0]
            second_best_class, second_best_sim = sorted_classes[1]
            similarity_gap = best_sim - second_best_sim

            # 若最佳與次佳類別相似度差距大於0.1，採用最佳結果
            if similarity_gap > 0.1:
                most_similar_class = best_class
                prediction_confidence = "中可信度-採用"
                prediction_reason = f"類別間相似度差距 {similarity_gap:.4f} > 0.1 (最佳:{best_class}={best_sim:.4f}, 次佳:{second_best_class}={second_best_sim:.4f})"
            else:
                most_similar_class = "未知類別"
                prediction_confidence = "中可信度-拒絕"
                prediction_reason = f"類別間相似度差距 {similarity_gap:.4f} <= 0.1 (最佳:{best_class}={best_sim:.4f}, 次佳:{second_best_class}={second_best_sim:.4f})"

    # 建立結果
    result = {
        'test_image': test_img_name,
        'class_name': class_name,
        'most_similar_class': most_similar_class,
        'similarity': highest_similarity,
        'confidence': prediction_confidence,
        'reason': prediction_reason
    }

    # log 輸出
    class_info = f" (類別: {class_name})" if class_name else ""
    log_to_file(log_file, f"輸入圖片：{test_img_name}{class_info}")
    log_to_file(log_file, f"判斷類別: {most_similar_class} ") # 如果你只要輸出類別的話把其他的註解掉就好囉
    log_to_file(log_file, f"相似度: {highest_similarity:.4f}, {prediction_confidence}")
    log_to_file(log_file, f"判斷依據: {prediction_reason}")
    log_to_file(log_file, f"========== 比對結束 {timestamp} ==========\n")

    return result


if __name__ == "__main__":
    # 👉 直接使用你提供的特徵向量
    # feature_vector = np.array([
    #     -0.13854486, -0.5864967, -0.44961542, 0.5388421,
    #     -0.15340725, -0.15852293, -0.06656379, 0.048497245,
    #     0.44523564, 0.08428633, -0.043040268, 0.16334336,
    #     0.11336197, 0.12026206, 0.51361555, -0.27661493,
    #     -0.19397274, 0.12098408, -0.48743692, -0.42168364,
    #     -0.055092317, -0.41428757, -0.34551904, -0.7258892,
    #     -0.029224703, 0.7987219, 0.23688623, -0.65080976,
    #     0.0680325, -0.02399005, -0.19103776, -0.07362068,
    #     -0.24717769, -0.1679785, -0.33343464, 0.4719361,
    #     -0.5086048, -0.28206697, 0.14941566, -0.82178366,
    #     0.26452893, 0.11221003, -0.45227346, 0.6749412,
    #     0.06695522, -0.6093571, 0.91329324, 0.069245175,
    #     -0.17280644, 0.20380846, 0.025364168, -0.24857843,
    #     -0.3983537, 0.6852318, -0.30120102, -0.8888354,
    #     0.64211375, 0.56690276, -0.008341664, -0.30756363,
    #     -0.78389573, -0.5504224, 0.15995675, -0.15108097
    # ], dtype=np.float32)

    feature_vector = np.array([
        -1.4695204e+00, -6.1041600e-01, -6.3751274e-01,  8.6774486e-01,
        -2.8947118e-01, -6.8347198e-01, -1.8429366e-01, -5.1768064e-01,
        1.7752104e+00, -1.0443141e+00, -2.8096586e-01,  2.2967619e-01,
        1.4362332e+00,  8.9772505e-01,  7.5321972e-01, -8.4329540e-01,
        -5.5402458e-01, -1.1753281e+00, -2.2896526e+00, -1.5719123e+00,
        -2.8654906e-01, -1.1622961e+00, -1.5807864e-01,  2.9897800e-01,
        4.9146256e-01, -1.6191038e+00,  9.2986530e-01,  1.2743329e-01,
        -1.3400313e-01, -1.2398878e+00, -4.6068197e-01, -8.9387518e-01,
        -4.7008300e-01,  2.6365042e-01, -1.0588495e+00, -5.0422895e-01,
        -1.3349142e+00,  1.3157506e+00, -6.0072867e-03, -9.7476524e-01,
        3.4556946e-01,  4.5259804e-02, -6.3122928e-01,  1.1429211e+00,
        -1.5074830e+00,  1.7644480e-01, -1.6441997e+00,  1.0451773e+00,
        1.0009202e+00,  4.4408354e-01, -1.5843607e+00, -7.8485870e-01,
        -7.0419538e-01,  2.1052783e-02, -6.0741687e-01,  1.3763286e-01,
        8.9226931e-01, -1.9781638e+00, -5.9097242e-01,  1.7636370e-02,
        2.4441257e+00, -1.5427278e-01,  1.8149583e-01, -2.8115240e-01,
        4.0551618e-01, -5.3443432e-01, -3.6272031e-01,  7.2740412e-01,
        -8.6835340e-02,  5.1878339e-01, -1.6919171e+00, -3.3152900e+00,
        3.1330854e-01, -4.5148426e-01,  6.4268209e-02,  2.9503283e-01,
        6.8297338e-01, -3.5677217e-02, -8.5037637e-01, -8.6635679e-01,
        -9.0656275e-01,  2.3354429e-01,  2.0188003e+00, -1.2048066e-01,
        1.8737234e+00,  1.4208332e+00,  9.0143704e-01, -8.0313021e-01,
        9.2001826e-01,  8.8673526e-01,  1.6149085e+00,  1.4966887e+00,
        -1.8464850e+00, -2.4569286e-02,  5.8916980e-01,  7.7232438e-01,
        -2.0152529e-01,  1.7579004e-01, -1.3836521e+00,  1.7451061e-01,
        -6.3706338e-01, -3.2270721e-01, -1.6942583e+00,  5.6751436e-01,
        3.9628023e-01, -1.8744233e+00,  1.6389489e+00,  7.3142159e-01,
        -3.4266615e+00, -2.9089339e+00, -2.1535601e-02,  2.2721633e-01,
        5.9249920e-01, -1.2323098e+00, -7.9150373e-01, -1.4483074e+00,
        2.1662889e-03, -1.3329610e+00, -6.6045654e-01,  7.7427375e-01,
        2.4077202e-01, -8.2813728e-01,  2.3518144e-01,  1.5156801e-01,
        6.6846931e-01, -1.6919734e+00,  2.2459890e-01, -2.0810546e-01
    ], dtype=np.float32)

    # './train_features.db'得在該資料夾中執行才有效
    db_file = './train_features.db'

    # result_dir = './output'
    # os.makedirs(result_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_file = os.path.join(result_dir, f'predict_from_vector_{timestamp}.txt')

    # print(f"\n執行特徵向量比對...\n結果將保存到: {log_file}")
    print(f"\n執行特徵向量比對...: ")
    predict_from_feature_vector(feature_vector, db_file, log_file=None)


