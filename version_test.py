import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 OneDNN 優化

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import numpy as np
import easyocr
import torch
import cv2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from matplotlib import font_manager
from sklearn.utils import resample
from sklearn.manifold import TSNE
from PIL import Image

# 常量定義
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 256  # 增大 batch size 以適應對比學習，減少 batch size 避免記憶體不足
TEMPERATURE = 0.1  # SimCLR 需要的溫度參數
DISTANCE_THRESHOLD = 1.4  # 用於判斷未知類別的距離閾值
TEXT_THRESHOLD = 0.8
LOW_TEXT = 0.3

# 設置中文字體
font_path = 'C:/Windows/Fonts/kaiu.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['DFKai-SB']
plt.rcParams['axes.unicode_minus'] = False


def gaussian_blur_opencv(image, kernel_size=3, sigma=0.3):
    image = np.array(image)  # 轉換成 NumPy 陣列
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

# 定義資料增強
def simclr_augment(image):
    image = tf.image.random_crop(image, size=[IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.05)  # 進一步降低
    image = tf.image.random_contrast(image, lower=0.95, upper=1.05)  # 進一步減小範圍
    image = tf.image.random_saturation(image, lower=0.95, upper=1.05)
    image = tf.image.random_hue(image, max_delta=0.02)  # 進一步降低
    if tf.random.uniform([]) > 0.5:
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # 移除噪聲和高斯模糊，避免過強增強
    return image

# 定義投影頭 (Projection Head) g(⋅)
def projection_head(x):
    x = Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dense(128, activation=None)(x)
    x = K.l2_normalize(x, axis=1)
    return x

# 修改生成正樣本對函數，加入 Hard Negative Mixing
def generate_positive_pairs(generator, feature_extractor, hard_negative_ratio=0.05, min_hard_negatives=2):
    while True:
        batch_x, batch_y = next(generator)
        batch_size = len(batch_x) // 2

        aug_1 = np.array([simclr_augment(img) for img in batch_x[:batch_size]])
        aug_2 = np.array([simclr_augment(img) for img in batch_x[:batch_size]])
        combined_input = np.concatenate([aug_1, aug_2], axis=0)

        features = feature_extractor.predict(combined_input, verbose=0)
        features = K.l2_normalize(features, axis=1).numpy()

        dist_matrix = np.linalg.norm(features[:, np.newaxis] - features[np.newaxis, :], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)

        hard_negatives = []
        for i in range(batch_size):
            distances = dist_matrix[i, batch_size:]
            num_hard_negatives = max(min_hard_negatives, int(batch_size * hard_negative_ratio))
            num_hard_negatives = min(num_hard_negatives, len(distances))
            hard_neg_idx = np.argsort(distances)[:num_hard_negatives]
            hard_neg_samples = aug_2[hard_neg_idx]

            if len(hard_neg_samples) >= 2:
                for _ in range(len(hard_neg_idx)):
                    idx1, idx2 = np.random.choice(len(hard_neg_samples), 2, replace=False)
                    alpha = np.random.uniform(0.5, 0.7)  # 調整範圍
                    new_neg = alpha * hard_neg_samples[idx1] + (1 - alpha) * hard_neg_samples[idx2]
                    hard_negatives.append(new_neg)

        if hard_negatives:
            hard_negatives = np.array(hard_negatives)
            combined_input = np.concatenate([combined_input, hard_negatives], axis=0)

        yield combined_input, batch_y


# 定義 NT-Xent Loss
def nt_xent_loss(y_true, y_pred):
    batch_size = tf.shape(y_pred)[0] // 2
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarity_matrix = tf.matmul(y_pred, y_pred, transpose_b=True) / TEMPERATURE
    mask = tf.eye(2 * batch_size)
    similarity_matrix = tf.where(mask == 1, -1e9, similarity_matrix)
    labels = tf.concat([tf.range(batch_size, 2 * batch_size), tf.range(batch_size)], axis=0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, similarity_matrix, from_logits=True)
    return tf.reduce_mean(loss)

def preprocess_input_simclr(image):
    image = simclr_augment(image)
    return image

# 建立 MobileNetV2 作為基礎特徵提取
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:50]:
    layer.trainable = False

# 定義 SimCLR 模型
input_tensor = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
features = base_model(input_tensor)
pooled_output = GlobalAveragePooling2D()(features)
projection_output = projection_head(pooled_output)
simclr_model = Model(inputs=input_tensor, outputs=projection_output)

# 提前創建 feature_extractor 用於 Hard Negative Mixing
feature_extractor = Model(inputs=simclr_model.input, outputs=simclr_model.output)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=100,
    decay_rate=0.9
)

simclr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=nt_xent_loss)

contrastive_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input_simclr,
    brightness_range=[0.95, 1.05],
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05
)

train_generator = contrastive_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE * 2,  # 雙倍 batch 以包含正樣本對
    class_mode='categorical',
    shuffle=True
)

def contrastive_generator(generator):
    return generate_positive_pairs(generator, feature_extractor, hard_negative_ratio=0.05, min_hard_negatives=2)

def limit_few_shot_samples(generator, n_samples=3):
    class_data = {cls: [] for cls in generator.class_indices.keys()}
    for _ in range(2):
        x, y = next(generator)
        for i in range(len(y)):
            cls = list(generator.class_indices.keys())[np.argmax(y[i])]
            class_data[cls].append(x[i])
    limited_data = []
    limited_labels = []
    for cls, data in class_data.items():
        sampled_data = resample(data, n_samples=min(n_samples, len(data)), random_state=42)
        limited_data.extend(sampled_data)
        limited_labels.extend([generator.class_indices[cls]] * len(sampled_data))
    return np.array(limited_data), np.array(limited_labels)

limited_data, limited_labels = limit_few_shot_samples(train_generator, n_samples=3)


# 替換 FewShotGenerator 類為生成器函數
def few_shot_generator(data, labels, batch_size):
    data = np.array(data)
    labels = to_categorical(labels)
    num_samples = len(data)
    while True:
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_x = data[batch_indices]
            batch_y = labels[batch_indices]
            yield batch_x, batch_y

few_shot_gen = few_shot_generator(limited_data, limited_labels, BATCH_SIZE * 2)

# 訓練 SimCLR 模型
history = simclr_model.fit(
    contrastive_generator(few_shot_gen),
    steps_per_epoch=max(1, len(limited_data) // (BATCH_SIZE * 2)),
    epochs=1000,  # 增加 epoch
    verbose=1
)
print("最終損失:", history.history['loss'][-1])

# 提取訓練數據的特徵
feature_extractor = Model(inputs=simclr_model.input, outputs=simclr_model.output)
features = feature_extractor.predict(limited_data, verbose=0)
labels = limited_labels
class_indices = train_generator.class_indices
num_classes = len(class_indices)

# 正規化原型
prototypes = []
for c in range(num_classes):
    class_features = features[labels == c]
    if len(class_features) > 0:
        mean_feat = np.mean(class_features, axis=0)
        mean_feat += 2e-3 * np.random.normal(size=mean_feat.shape)
        norm_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-10)
    else:
        norm_feat = np.zeros(features.shape[1])
    prototypes.append(norm_feat)
prototypes = np.array(prototypes)

class_names = {v: k for k, v in class_indices.items()}
for c in range(num_classes):
    class_feats = features[labels == c]
    if len(class_feats) > 0:
        avg_dist = np.mean(np.linalg.norm(class_feats - prototypes[c], axis=1))
        print(f"類別 {class_names[c]} 的平均距離: {avg_dist:.2f}")

# 特徵提取（加強正規化檢查）
def extract_features(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    feat = feature_extractor.predict(img_array, verbose=0)
    norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-10
    return feat / norm

batch_x, _ = next(train_generator)
aug_1 = feature_extractor.predict(np.array([simclr_augment(img) for img in limited_data]))
aug_2 = feature_extractor.predict(np.array([simclr_augment(img) for img in limited_data]))
pos_dist = np.mean(np.linalg.norm(aug_1 - aug_2, axis=1))
print(f"正樣本平均距離: {pos_dist:.2f}")
dist_matrix = np.linalg.norm(aug_1[:, np.newaxis] - aug_1[np.newaxis, :], axis=2)
np.fill_diagonal(dist_matrix, np.inf)
neg_dist = np.mean(dist_matrix[dist_matrix != np.inf])
print(f"負樣本平均距離: {neg_dist:.2f}")

# 預處理圖片以進行 OCR
def preprocess_image_for_ocr(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"無法載入圖片: {image_path}")
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return Image.fromarray(gray)
    except Exception as e:
        print(f"預處理錯誤: {e}")
        return None

# 測試範例函數
def test_example(image_path):
    try:
        test_feature = extract_features(image_path)
        distances = np.linalg.norm(prototypes - test_feature, axis=1)
        mean_distance = np.mean([np.min(np.linalg.norm(prototypes - p, axis=1)) for p in prototypes])
        adaptive_threshold = (pos_dist + neg_dist) / 2
        print(f"Mean Distance: {mean_distance:.2f}, Adaptive Threshold: {adaptive_threshold:.2f}")

        class_names = {v: k for k, v in class_indices.items()}
        for c in range(num_classes):
            class_feats = features[labels == c]
            if len(class_feats) > 0:
                avg_dist = np.mean(np.linalg.norm(class_feats - prototypes[c], axis=1))
                print(f"類別 {class_names[c]} 的平均距離: {avg_dist:.2f}")

        if np.min(distances) > adaptive_threshold:
            predicted_class = "Unknown"
        else:
            predicted_class_idx = np.argmin(distances)
            predicted_class = class_names[predicted_class_idx]

        for idx, dist in enumerate(distances):
            print(f"與 {class_names[idx]} 的距離: {dist:.2f}")
        print(f"初步預測類別: {predicted_class}")

        # OCR 部分
        processed_img = preprocess_image_for_ocr(image_path)
        if processed_img is None:
            return predicted_class, distances, [], []

        reader = easyocr.Reader(['ch_tra'], gpu=torch.cuda.is_available())
        results = {
            "原圖": reader.readtext(image_path, detail=1, text_threshold=TEXT_THRESHOLD, low_text=LOW_TEXT),
            "預處理後": reader.readtext(np.array(processed_img), detail=1, text_threshold=TEXT_THRESHOLD, low_text=LOW_TEXT)
        }

        for mode, text in results.items():
            print(f"EasyOCR ({mode}) 提取的文字: {[res[1] for res in text]}")

        # OCR 關鍵詞調整
        adjustments = {
            "祥豐": ("bus_stop_xiang_feng", 1.0, ["祥豐", "豐"]),
            "自由": ("ntou_freedom_ship", 1.0, ["自由", "由自", "自"]),
            "體育館": ("bus_stop_gymnasium", 1.0, ["體育館", "育館", "館"]),
            "甜甜圈": ("ntou_donut", 1.0, ["甜甜圈", "甜圈"]),
            "影印": ("bus_stop_xiang_feng", 1.0, ["影印", "印"])
        }

        detected_scores = {}
        all_texts = [res[1] for res in results["原圖"] + results["預處理後"]]

        for keyword, (new_class, weight, keyword_variants) in adjustments.items():
            if any(any(variant in text for variant in keyword_variants) for text in all_texts):
                detected_scores[new_class] = detected_scores.get(new_class, 0) + weight
                print(f"檢測到 {keyword}（或其變形），提升 {new_class} 可能性 (權重: {weight})")

        # 以 OCR 解析結果加權選擇最終類別
        if detected_scores:
            max_score = max(detected_scores.values())
            top_classes = [k for k, v in detected_scores.items() if v == max_score]
            if len(top_classes) > 1:
                final_predicted_class = "Unknown"
                print(f"多個關鍵詞權重相等，選擇 Unknown")
            else:
                final_scores = {
                    k: (distances[class_indices[k]] * 0.2 + (1 - detected_scores[k] / sum(detected_scores.values())) * 0.8)
                    for k in detected_scores
                }
                final_predicted_class = min(final_scores, key=final_scores.get)
                print(f"基於 OCR 選擇類別: {final_predicted_class}")
        elif np.min(distances) > adaptive_threshold:
            final_predicted_class = "Unknown"
        else:
            final_predicted_class = class_names[np.argmin(distances)]

        print(f"結合 OCR 後的最終預測類別: {final_predicted_class}")
        return final_predicted_class, distances, results["原圖"], results["預處理後"]
    except Exception as e:
        print(f"測試錯誤: {e}")
        return "Error", [], [], []


# 主程式
if __name__ == "__main__":
    test_images = ['test.jpg', 'test1.jpg', 'test2.jpg', 'test3.jpg']
    test_features = []
    test_labels = []
    for image_path in test_images:
        print(f"處理圖片: {image_path}")
        final_predicted_class, _, result_1, result_2 = test_example(image_path)
        test_feature = extract_features(image_path)
        test_features.append(test_feature.flatten())
        test_labels.append(class_indices.get(final_predicted_class, -1))
        print("-" * 50)

    all_features = np.vstack([features, np.array(test_features)])
    all_labels = np.concatenate([labels, np.array(test_labels)])
    valid_idx = all_labels != -1
    filtered_features = all_features[valid_idx]
    filtered_labels = all_labels[valid_idx]

    print(f"總 Few-shot 樣本數: {len(limited_data)}, 標籤數: {len(limited_labels)}")
    from collections import Counter
    print(Counter(limited_labels))

    # perplexity 困惑度、early exaggeration factor 前期放大係數
    # learning rate 學習率、maximum number of iterations 最大迭代次數、angle 角度
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, learning_rate=20)
    tsne_results = tsne.fit_transform(filtered_features)

    plt.figure(figsize=(10, 8))
    class_names = {v: k for k, v in class_indices.items()}

    # 繪製訓練數據
    colors = ['blue', 'orange', 'green', 'red']
    for c in range(num_classes):
        idx = filtered_labels == c
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=colors[c], label=class_names[c], alpha=0.4, s=80)

    # 繪製測試數據
    markers = ['*', 's', 'o', '^']
    test_start_idx = len(labels)
    filtered_test_indices = np.where(valid_idx[test_start_idx:])[0]
    for i, idx in enumerate(filtered_test_indices):
        plt.scatter(tsne_results[test_start_idx + idx, 0], tsne_results[test_start_idx + idx, 1],
                    c='black', marker=markers[i % len(markers)], s=250, label=f'{test_images[idx]}',
                    edgecolors='white', linewidth=1.5)

    plt.legend()
    plt.title('t-SNE 視覺化：訓練與測試數據特徵分佈')
    plt.xlabel('t-SNE 維度 1')
    plt.ylabel('t-SNE 維度 2')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()