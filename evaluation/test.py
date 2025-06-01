import pickle
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

# 1. Load dữ liệu đã tiền xử lý
with open('preprocessing/processed_data_fixed.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test, label_encoder = pickle.load(f)

# Kiểm tra kích thước dữ liệu
num_samples_train, timesteps, num_features = X_train.shape
num_samples_test = X_test.shape[0]
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Chuyển đổi nhãn thành tên hành động
action_labels = label_encoder.classes_  # ['standing', 'walking', 'jogging', 'sitting', 'falling', 'jumping']
y_test_labels = [action_labels[i] for i in y_test]

# 2. Load mô hình đã huấn luyện
model = load_model('cnn_lstm.h5')
print("✅ Đã load mô hình đã huấn luyện")

# 3. Dự đoán hành động trên tập test
y_pred = model.predict(X_test, verbose=1)
y_pred_labels = [action_labels[np.argmax(pred)] for pred in y_pred]

# 4. Tạo DataFrame từ dữ liệu test
# Giả sử mỗi mẫu trong X_test có các cột: accelX, accelY, accelZ, gyroX, gyroY, gyroZ
# Tạo timestamp giả (tăng dần từ 0)
timestamps = np.arange(len(X_test)) * timesteps / 50.0  # Giả sử tần số 50Hz

# Tạo DataFrame
data = pd.DataFrame({
    'timestamp': timestamps,
    'accelX': X_test[:, :, 0].mean(axis=1),  # Trung bình theo timesteps
    'accelY': X_test[:, :, 1].mean(axis=1),
    'accelZ': X_test[:, :, 2].mean(axis=1),
    'gyroX': X_test[:, :, 3].mean(axis=1),
    'gyroY': X_test[:, :, 4].mean(axis=1),
    'gyroZ': X_test[:, :, 5].mean(axis=1),
    'action': y_pred_labels
})

# 5. Visualize dữ liệu cảm biến
plt.figure(figsize=(15, 10))

# Gia tốc
plt.subplot(2, 1, 1)
plt.plot(data['timestamp'], data['accelX'], label='accelX', color='r')
plt.plot(data['timestamp'], data['accelY'], label='accelY', color='g')
plt.plot(data['timestamp'], data['accelZ'], label='accelZ', color='b')
plt.title('Gia tốc theo thời gian (trung bình trên mỗi mẫu)')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Gia tốc (m/s²)')
plt.legend()
plt.grid(True)

# Con quay
plt.subplot(2, 1, 2)
plt.plot(data['timestamp'], data['gyroX'], label='gyroX', color='r')
plt.plot(data['timestamp'], data['gyroY'], label='gyroY', color='g')
plt.plot(data['timestamp'], data['gyroZ'], label='gyroZ', color='b')
plt.title('Con quay theo thời gian (trung bình trên mỗi mẫu)')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Tốc độ góc (rad/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Visualize hành động dự đoán
plt.figure(figsize=(15, 5))
data['action_numeric'] = data['action'].apply(lambda x: action_labels.tolist().index(x))
plt.scatter(data['timestamp'], data['action_numeric'], c=data['action_numeric'], cmap='viridis', s=50)
plt.yticks(range(len(action_labels)), action_labels)
plt.title('Hành động dự đoán theo thời gian')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Hành động')
plt.grid(True)
plt.show()

# 7. Visualize phân bố hành động
plt.figure(figsize=(10, 5))
sns.countplot(x='action', data=data, order=action_labels)
plt.title('Phân bố các hành động dự đoán')
plt.xlabel('Hành động')
plt.ylabel('Số lần xuất hiện')
plt.xticks(rotation=45)
plt.show()

# 8. Trích xuất đặc trưng từ lớp ẩn của mô hình
feature_extractor = Model(inputs=model.input, outputs=model.layers[2].output)  # Lớp LSTM thứ hai
features = feature_extractor.predict(X_test)
print(f"Kích thước đặc trưng: {features.shape}")

# 9. Giảm chiều dữ liệu bằng t-SNE
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)
print(f"Kích thước đặc trưng sau t-SNE: {features_2d.shape}")

# Tạo DataFrame để visualize đặc trưng
df_features = pd.DataFrame({
    'x': features_2d[:, 0],
    'y': features_2d[:, 1],
    'action': y_test_labels
})

# Visualize đặc trưng bằng t-SNE
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_features, x='x', y='y', hue='action', style='action', palette='deep', s=100)
plt.title('Đặc trưng của 6 hành động (t-SNE)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Hành động')
plt.grid(True)
plt.show()

# 10. Giảm chiều bằng PCA để so sánh
pca = PCA(n_components=2)
features_2d_pca = pca.fit_transform(features)
df_pca = pd.DataFrame({
    'x': features_2d_pca[:, 0],
    'y': features_2d_pca[:, 1],
    'action': y_test_labels
})

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca, x='x', y='y', hue='action', style='action', palette='deep', s=100)
plt.title('Đặc trưng của 6 hành động (PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Hành động')
plt.grid(True)
plt.show()

# 11. Visualize giá trị trung bình của đặc trưng theo hành động
features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
features_df['action'] = y_test_labels
mean_features = features_df.groupby('action').mean()

plt.figure(figsize=(12, 6))
sns.heatmap(mean_features, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Giá trị trung bình của đặc trưng theo hành động')
plt.xlabel('Đặc trưng')
plt.ylabel('Hành động')
plt.show()