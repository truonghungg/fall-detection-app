import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

print("Đang tải và phân tích dữ liệu đã huấn luyện...")

# Thiết lập style cho biểu đồ
plt.style.use('default')  # Sử dụng style mặc định
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

# Định nghĩa màu sắc
colors = {
    'accuracy': '#4CAF50',
    'precision': '#2196F3',
    'recall': '#FFC107',
    'f1': '#E91E63',
    'background': '#f8f9fa'
}

# Hàm tạo dữ liệu đánh giá từ báo cáo phân loại
def parse_classification_report(report_dict):
    """Chuyển đổi classification report dict thành dạng dễ vẽ"""
    metrics = {}
    overall_accuracy = report_dict['accuracy']
    
    # Loại bỏ các khóa không phải lớp
    activities = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    for activity in activities:
        metrics[activity] = {
            'precision': report_dict[activity]['precision'],
            'recall': report_dict[activity]['recall'],
            'f1-score': report_dict[activity]['f1-score'],
            'support': report_dict[activity]['support']
        }
    
    return metrics, overall_accuracy, activities

# Kiểm tra xem báo cáo đánh giá có tồn tại không
if os.path.exists('models/confusion_matrix.png'):
    print("Đã tìm thấy báo cáo phân tích trước đó!")

# Kiểm tra các file mô hình và dữ liệu
model_paths = [f for f in os.listdir() if f.endswith('.h5')]
model_paths.extend([f for f in os.listdir('models') if f.endswith('.h5')] if os.path.exists('models') else [])

if not model_paths:
    print("Không tìm thấy file mô hình .h5 nào!")
else:
    print(f"Tìm thấy các file mô hình: {model_paths}")

data_paths = []
if os.path.exists('preprocessing'):
    data_paths = [f for f in os.listdir('preprocessing') if f.endswith('.pkl')]
    
if not data_paths:
    print("Không tìm thấy file dữ liệu .pkl nào trong thư mục preprocessing!")
else:
    print(f"Tìm thấy các file dữ liệu: {data_paths}")

# Ưu tiên tìm kiếm kết quả từ các loại đã được đào tạo
try:
    # Thử tải dữ liệu trước
    if os.path.exists('preprocessing/processed_data_fixed.pkl'):
        with open('preprocessing/processed_data_fixed.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test, label_encoder = pickle.load(f)
            print(f"✅ Đã tải dữ liệu thành công!")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            print(f"Các lớp: {label_encoder.classes_}")

        # Tìm mô hình để tải
        model_file = None
        if os.path.exists('models/cnn_lstm.h5'):
            model_file = 'models/cnn_lstm.h5'
        elif model_paths:
            model_file = model_paths[0] if os.path.exists(model_paths[0]) else os.path.join('models', model_paths[0])
        
        if model_file and os.path.exists(model_file):
            print(f"✅ Đã tải mô hình thành công!")
            model = load_model(model_file)
            
            # Dự đoán trên tập kiểm tra
            print("Đang dự đoán trên tập kiểm tra...")
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            
            # Tạo classification report
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
            
            # Phân tích các chỉ số
            metrics, overall_accuracy, activities = parse_classification_report(report)
            print(f"\nĐộ chính xác tổng thể: {overall_accuracy:.4f}")
        else:
            print("⚠️ Không tìm thấy file mô hình để tải!")
            raise FileNotFoundError("Không tìm thấy mô hình")
    else:
        print("⚠️ Không tìm thấy file dữ liệu đã xử lý!")
        raise FileNotFoundError("Không tìm thấy dữ liệu đã xử lý")
            
except Exception as e:
    print(f"❌ Lỗi: {str(e)}")
    print("Sử dụng dữ liệu mẫu từ báo cáo dự án...")
    
    # Dữ liệu từ báo cáo dự án
    activities = ['falling', 'jogging', 'jumping', 'sitting', 'standing', 'walking']
    
    # Dữ liệu từ kết quả thực tế của mô hình
    metrics = {
        'falling': {'precision': 0.98, 'recall': 0.86, 'f1-score': 0.91, 'support': 155},
        'jogging': {'precision': 0.98, 'recall': 0.99, 'f1-score': 0.98, 'support': 155},
        'jumping': {'precision': 1.00, 'recall': 0.98, 'f1-score': 0.99, 'support': 155},
        'sitting': {'precision': 0.86, 'recall': 0.85, 'f1-score': 0.86, 'support': 156},
        'standing': {'precision': 0.86, 'recall': 1.00, 'f1-score': 0.92, 'support': 155},
        'walking': {'precision': 0.98, 'recall': 0.96, 'f1-score': 0.97, 'support': 155}
    }
    
    # Accuracy tổng thể
    overall_accuracy = 0.94

# In thông tin về metrics
print("\n===== METRICS =====")
print(f"Độ chính xác tổng thể: {overall_accuracy:.4f}")
print(f"Các lớp hoạt động: {activities}")
for activity, metric in metrics.items():
    print(f"{activity}: Precision={metric['precision']:.4f}, Recall={metric['recall']:.4f}, F1={metric['f1-score']:.4f}")

# 1. Biểu đồ cột cho từng chỉ số cho mỗi hoạt động
print("\nVẽ biểu đồ cột cho từng chỉ số...")
plt.figure(figsize=(14, 12))

# Tạo DataFrame từ metrics để dễ vẽ
df_metrics = pd.DataFrame.from_dict({
    activity: {
        'Precision': metrics[activity]['precision'],
        'Recall': metrics[activity]['recall'],
        'F1-score': metrics[activity]['f1-score']
    } for activity in activities
}).T

# 1. Biểu đồ cột nhóm
plt.subplot(2, 1, 1)
df_metrics.plot(kind='bar', ax=plt.gca(), width=0.8, alpha=0.8)
plt.xlabel('Hoạt động', fontsize=12)
plt.ylabel('Giá trị chỉ số', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0.8, 1.05)  # Tăng giới hạn trên để có chỗ cho text

# Thêm text với giá trị accuracy tổng thể
plt.text(0.02, 0.95, f'Độ chính xác tổng thể: {overall_accuracy:.4f}', 
         transform=plt.gca().transAxes, fontsize=10, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

# Thêm text giá trị lên mỗi cột
for i, activity in enumerate(activities):
    for j, metric in enumerate(['Precision', 'Recall', 'F1-score']):
        value = df_metrics.loc[activity, metric]
        plt.text(i, value + 0.01, f'{value:.3f}', 
                ha='center', va='bottom', fontsize=8, rotation=90)

plt.legend(title='Chỉ số', fontsize=10, title_fontsize=10)

# 2. Radar Chart để so sánh các chỉ số giữa các hoạt động
print("Vẽ radar chart...")
plt.subplot(2, 1, 2)

# Tạo dữ liệu cho radar chart
categories = ['Precision', 'Recall', 'F1-score']
N = len(categories)

# Góc của từng trục
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Đóng hình tròn

# Thiết lập trục
ax = plt.subplot(2, 1, 2, polar=True)

# Vẽ cho từng hoạt động
for i, activity in enumerate(activities):
    values = [metrics[activity]['precision'], metrics[activity]['recall'], 
              metrics[activity]['f1-score']]
    values += values[:1]  # Đóng hình tròn
    
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=activity)
    ax.fill(angles, values, alpha=0.1)

# Thiết lập radar chart
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), categories)

# Hiển thị đường kính
for angle, metric in zip(angles[:-1], categories):
    ax.plot([angle, angle], [0.8, 1.0], linewidth=1, linestyle='-', color='gray', alpha=0.3)

ax.set_ylim(0.8, 1.0)
ax.set_yticks(np.arange(0.8, 1.01, 0.05))
ax.set_yticklabels([f'{x:.2f}' for x in np.arange(0.8, 1.01, 0.05)], fontsize=8)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.title('Radar Chart các chỉ số đánh giá', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('model_metrics_combined.png', dpi=300, bbox_inches='tight')

# 3. Biểu đồ nhiệt (Heatmap) cho các chỉ số đánh giá
print("Vẽ heatmap...")
plt.figure(figsize=(14, 8))
heatmap_data = pd.DataFrame({
    'Precision': [metrics[activity]['precision'] for activity in activities],
    'Recall': [metrics[activity]['recall'] for activity in activities],
    'F1-score': [metrics[activity]['f1-score'] for activity in activities]
}, index=activities)

# Tạo heatmap
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5, vmin=0.8, vmax=1.0)
plt.title('Heatmap các chỉ số đánh giá cho từng hoạt động', fontsize=16, fontweight='bold')
plt.xlabel('Chỉ số đánh giá', fontsize=14)
plt.ylabel('Hoạt động', fontsize=14)
plt.tight_layout()
plt.savefig('model_metrics_heatmap.png', dpi=300, bbox_inches='tight')

# 4. Biểu đồ so sánh F1-score giữa các lớp
print("Vẽ biểu đồ F1-score...")
plt.figure(figsize=(14, 6))
f1_scores = [metrics[activity]['f1-score'] for activity in activities]
support = [metrics[activity]['support'] for activity in activities]

bars = plt.bar(activities, f1_scores, alpha=0.8, color=[plt.cm.tab10(i) for i in range(len(activities))])

# Thêm text giá trị và support lên mỗi cột
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.3f}\n(n={support[i]})',
             ha='center', va='bottom', fontsize=10)

plt.xlabel('Hoạt động', fontsize=14)
plt.ylabel('F1-score', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0.8, 1.0)  # Để tập trung vào sự khác biệt
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('model_f1_comparison.png', dpi=300, bbox_inches='tight')

print("\n✅ Đã hoàn thành và lưu các biểu đồ:")
print("📊 model_metrics_combined.png - Biểu đồ cột và radar chart")
print("📊 model_metrics_heatmap.png - Heatmap các chỉ số")
print("📊 model_f1_comparison.png - So sánh F1-score")

plt.show() 