import os
import datetime
import numpy as np
import sqlite3

def log_to_file(log_file, message):
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    print(message)

def load_features_from_database(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, label, feature FROM features")

    file_paths = []
    labels = []
    features = []

    for row in cursor.fetchall():
        path, label, feature_blob = row
        feature_array = np.frombuffer(feature_blob, dtype=np.float32)
        file_paths.append(path)
        labels.append(label)
        features.append(feature_array)

    conn.close()
    return features, file_paths, labels

def predict_from_feature_vector(feature_vector, top_k=5, db_file='train_features.db', log_file=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(log_file, f"\n========== 特徵向量比對開始 {timestamp} ==========\n")

    train_features, train_file_paths, train_labels = load_features_from_database(db_file)
    if len(train_features) == 0:
        log_to_file(log_file, "訓練資料特徵為空，無法進行比對")
        return []

    similarities = []
    for idx, train_feature in enumerate(train_features):
        similarity = np.dot(feature_vector, train_feature) / (
            np.linalg.norm(feature_vector) * np.linalg.norm(train_feature)
        )
        similarities.append((similarity, train_file_paths[idx], train_labels[idx]))

    similarities.sort(reverse=True, key=lambda x: x[0])
    top_matches = similarities[:top_k]

    for rank, (sim, path, label) in enumerate(top_matches, start=1):
        log_to_file(log_file, f"Top {rank}: 類別={label}, 相似度={sim:.4f}, 圖片={path}")

    log_to_file(log_file, f"\n========== 比對結束 {timestamp} ==========\n")
    return top_matches

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


    db_file = './train_features.db'
    # result_dir = './output'
    # os.makedirs(result_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_file = os.path.join(result_dir, f'predict_from_vector_{timestamp}.txt')

    top_k = 5
    # print(f"\n執行特徵向量比對...\n結果將保存到: {log_file}")
    print(f"\n執行特徵向量比對...: ")
    predict_from_feature_vector(feature_vector, top_k, db_file, log_file=None)
