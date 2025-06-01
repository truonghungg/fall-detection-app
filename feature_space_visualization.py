import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tqdm import tqdm  # Thêm thanh tiến độ

# Thiết lập style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 12

print("Đang tải dữ liệu để trực quan hóa không gian đặc trưng...")

# Màu sắc cho từng lớp hoạt động
colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6', '#e67e22']
markers = ['o', 's', '^', 'D', 'P', '*']

# Hàm tạo dữ liệu ngẫu nhiên nếu không có dữ liệu thực
def generate_sample_data():
    """Tạo dữ liệu mẫu cho trực quan hóa không gian đặc trưng"""
    print("Tạo dữ liệu mẫu với ranh giới phân lớp rõ ràng...")
    
    # Định nghĩa các hoạt động
    activities = ['Đi bộ', 'Chạy', 'Ngồi', 'Đứng', 'Nhảy', 'Té ngã']
    num_samples_per_class = 200
    num_features = 14
    
    # Tạo dữ liệu mẫu với ranh giới phân tách rõ ràng
    X = []
    y = []
    
    # Định nghĩa trung tâm các cụm cho các lớp trong không gian 14 chiều
    centers = {
        'Đi bộ': np.array([0.6, 0.3, 0.2, 0.1, 0.2, 0.3, 0.5, 0.3, 0.4, 0.2, 0.7, 0.2, 0.1, 0.3]),
        'Chạy': np.array([0.8, 0.7, 0.5, 0.4, 0.6, 0.7, 0.9, 0.8, 0.7, 0.5, 0.9, 0.3, 0.2, 0.5]),
        'Ngồi': np.array([0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.6, 0.5, 0.1]),
        'Đứng': np.array([0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.3, 0.1]),
        'Nhảy': np.array([0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.9, 0.8, 0.7, 0.5, 0.4, 0.4, 0.7]),
        'Té ngã': np.array([0.9, 0.9, 0.8, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.3, 0.7, 0.7, 0.9])
    }
    
    # Tạo dữ liệu cho từng lớp
    for i, activity in enumerate(activities):
        # Tạo dữ liệu ngẫu nhiên xung quanh trung tâm
        center = centers[activity]
        # Độ phân tán khác nhau cho từng lớp
        std_dev = 0.05 if activity in ['Ngồi', 'Đứng'] else 0.1
        
        for _ in range(num_samples_per_class):
            # Tạo điểm dữ liệu quanh tâm với phân phối Gaussian
            sample = center + np.random.normal(0, std_dev, num_features)
            # Giới hạn giá trị trong khoảng [0, 1]
            sample = np.clip(sample, 0, 1)
            X.append(sample)
            y.append(i)
    
    # Chuyển đổi thành mảng numpy
    X = np.array(X)
    y = np.array(y)
    
    return X, y, activities

# Tải dữ liệu thực tế hoặc tạo dữ liệu mẫu
try:
    # Kiểm tra xem có dữ liệu thực tế không
    if os.path.exists('preprocessing/processed_data.pkl'):
        print("Tìm thấy dữ liệu đã xử lý. Đang tải...")
        with open('preprocessing/processed_data.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test, label_encoder = pickle.load(f)
            
            # Lấy tên các hoạt động
            activities = label_encoder.classes_
            print(f"Các hoạt động: {activities}")
            
            # Chọn 1000 mẫu ngẫu nhiên từ tập huấn luyện để giảm thời gian tính toán
            if len(X_train) > 1000:
                print(f"Chọn mẫu 1000 điểm dữ liệu từ {len(X_train)} mẫu...")
                indices = np.random.choice(len(X_train), 1000, replace=False)
                X_sampled = X_train[indices]
                y_sampled = y_train[indices]
            else:
                X_sampled = X_train
                y_sampled = y_train
            
            # Trích xuất đặc trưng từ dữ liệu đã xử lý
            # Mỗi cửa sổ dữ liệu là một mảng 3D (window_size, num_features)
            # Cần chuyển thành vector đặc trưng 1D
            print(f"Lấy đặc trưng từ dữ liệu cửa sổ dạng {X_sampled.shape}...")
            
            # Tính toán giá trị trung bình của mỗi đặc trưng trên mỗi cửa sổ
            X_features = np.mean(X_sampled, axis=1)
            print(f"Kích thước đặc trưng sau khi lấy trung bình: {X_features.shape}")
            
            # Lấy nhãn
            y_labels = y_sampled
            
            print("✅ Đã tải xong dữ liệu thực tế!")
    else:
        print("Không tìm thấy dữ liệu thực tế. Sử dụng dữ liệu mẫu...")
        X_features, y_labels, activities = generate_sample_data()
except Exception as e:
    print(f"Lỗi khi tải dữ liệu: {str(e)}")
    print("Sử dụng dữ liệu mẫu...")
    X_features, y_labels, activities = generate_sample_data()

# Áp dụng PCA để giảm chiều xuống 2D
print("Áp dụng PCA giảm chiều...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_features)
print(f"Tỷ lệ phương sai giải thích: {pca.explained_variance_ratio_}")

# Áp dụng t-SNE để giảm chiều xuống 2D
print("Áp dụng t-SNE giảm chiều (quá trình này có thể mất vài phút)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_features) - 1))
X_tsne = tsne.fit_transform(X_features)

# Áp dụng PCA để giảm chiều xuống 3D
pca3d = PCA(n_components=3)
X_pca3d = pca3d.fit_transform(X_features)

# 1. Biểu đồ kết quả PCA 2D
plt.figure(figsize=(12, 10))

# Vẽ điểm dữ liệu với màu theo nhãn hoạt động
for i, activity in enumerate(activities):
    # Lấy các điểm thuộc hoạt động hiện tại
    indices = np.where(y_labels == i)[0]
    plt.scatter(
        X_pca[indices, 0], 
        X_pca[indices, 1],
        c=[colors[i]] * len(indices),
        label=activity,
        alpha=0.7,
        edgecolors='w',
        s=70,
        marker=markers[i % len(markers)]
    )

# Thêm tên trục và tiêu đề
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
plt.title('PCA: Phân bố các hoạt động trong không gian đặc trưng 2D', fontsize=16, fontweight='bold')

# Thêm grid và legend
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Lưu biểu đồ
plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')

# 2. Biểu đồ kết quả t-SNE 2D
plt.figure(figsize=(12, 10))

# Vẽ điểm dữ liệu với màu theo nhãn hoạt động
for i, activity in enumerate(activities):
    # Lấy các điểm thuộc hoạt động hiện tại
    indices = np.where(y_labels == i)[0]
    plt.scatter(
        X_tsne[indices, 0], 
        X_tsne[indices, 1],
        c=[colors[i]] * len(indices),
        label=activity,
        alpha=0.7,
        edgecolors='w',
        s=70,
        marker=markers[i % len(markers)]
    )

# Thêm tên trục và tiêu đề
plt.xlabel('t-SNE feature 1', fontsize=14)
plt.ylabel('t-SNE feature 2', fontsize=14)
plt.title('t-SNE: Phân bố các hoạt động trong không gian đặc trưng 2D', fontsize=16, fontweight='bold')

# Thêm grid và legend
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Lưu biểu đồ
plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')

# 3. Biểu đồ PCA 3D
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Vẽ điểm dữ liệu với màu theo nhãn hoạt động
for i, activity in enumerate(activities):
    # Lấy các điểm thuộc hoạt động hiện tại
    indices = np.where(y_labels == i)[0]
    ax.scatter(
        X_pca3d[indices, 0], 
        X_pca3d[indices, 1], 
        X_pca3d[indices, 2],
        c=[colors[i]] * len(indices),
        label=activity,
        alpha=0.7,
        edgecolors='w',
        s=70,
        marker=markers[i % len(markers)]
    )

# Thêm tên trục và tiêu đề
ax.set_xlabel(f'PC1 ({pca3d.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
ax.set_ylabel(f'PC2 ({pca3d.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
ax.set_zlabel(f'PC3 ({pca3d.explained_variance_ratio_[2]:.2%} variance)', fontsize=14)
plt.title('PCA 3D: Phân bố các hoạt động trong không gian đặc trưng', fontsize=16, fontweight='bold')

# Thêm legend
plt.legend(fontsize=12)
plt.tight_layout()

# Lưu biểu đồ
plt.savefig('pca3d_visualization.png', dpi=300, bbox_inches='tight')

# 4. Biểu đồ heatmap hiển thị tầm quan trọng của các đặc trưng trong PCA
plt.figure(figsize=(14, 8))

# Tạo DataFrame từ components của PCA
feature_names = [f'Feature {i+1}' for i in range(X_features.shape[1])]
components_df = pd.DataFrame(
    data=pca.components_,
    columns=feature_names
)

# Tạo heatmap
sns.heatmap(components_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Mức độ đóng góp của các đặc trưng trong PCA', fontsize=16, fontweight='bold')
plt.xlabel('Đặc trưng', fontsize=14)
plt.ylabel('Principal Components', fontsize=14)
plt.tight_layout()

# Lưu biểu đồ
plt.savefig('pca_features_heatmap.png', dpi=300, bbox_inches='tight')

# 5. Biểu đồ kết hợp PCA vs t-SNE
plt.figure(figsize=(18, 8))

# Subplot cho PCA
plt.subplot(1, 2, 1)
for i, activity in enumerate(activities):
    indices = np.where(y_labels == i)[0]
    plt.scatter(
        X_pca[indices, 0], 
        X_pca[indices, 1],
        c=[colors[i]] * len(indices),
        label=activity,
        alpha=0.7,
        s=70,
        marker=markers[i % len(markers)]
    )
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
plt.title('PCA', fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(fontsize=10)

# Subplot cho t-SNE
plt.subplot(1, 2, 2)
for i, activity in enumerate(activities):
    indices = np.where(y_labels == i)[0]
    plt.scatter(
        X_tsne[indices, 0], 
        X_tsne[indices, 1],
        c=[colors[i]] * len(indices),
        label=activity,
        alpha=0.7,
        s=70,
        marker=markers[i % len(markers)]
    )
plt.xlabel('t-SNE feature 1', fontsize=14)
plt.ylabel('t-SNE feature 2', fontsize=14)
plt.title('t-SNE', fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(fontsize=10)

plt.suptitle('So sánh phân bố hoạt động giữa PCA và t-SNE', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()

# Lưu biểu đồ
plt.savefig('pca_tsne_comparison.png', dpi=300, bbox_inches='tight')

print("✅ Đã tạo và lưu các biểu đồ trực quan hóa không gian đặc trưng:")
print("📊 pca_visualization.png - PCA 2D")
print("📊 tsne_visualization.png - t-SNE 2D")
print("📊 pca3d_visualization.png - PCA 3D")
print("📊 pca_features_heatmap.png - Heatmap các đặc trưng")
print("📊 pca_tsne_comparison.png - So sánh PCA và t-SNE")

plt.show() 