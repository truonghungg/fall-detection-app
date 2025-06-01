import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

def get_plot_title(filename):
    """Tạo tiêu đề đồ thị từ tên file"""
    # Tách thông tin từ tên file
    parts = filename.replace('.csv', '').split('_')
    condition = parts[0]  # tight/loose/normal
    movement = parts[1]   # static/rotation
    
    # Chuyển đổi điều kiện đeo
    condition_map = {
        'tight': 'Đeo chặt',
        'loose': 'Đeo lỏng',
        'normal': 'Đeo bình thường'
    }
    
    # Chuyển đổi chuyển động
    movement_map = {
        'static': 'Giữ nguyên',
        'rotation': 'Vung cánh tay từ trái sang phải'
    }
    
    return f"{condition_map.get(condition, condition)} - {movement_map.get(movement, movement)}"

def plot_sensor_data(file_path):
    """Vẽ đồ thị dữ liệu cảm biến từ file CSV"""
    # Đọc dữ liệu
    df = pd.read_csv(file_path)
    
    # Tạo figure với 6 subplot (2 hàng, 3 cột)
    fig, axes = plt.subplots(2, 3, figsize=(18, 14))
    fig.suptitle(get_plot_title(os.path.basename(file_path)), fontsize=16)
    
    # Làm phẳng mảng axes untuk dễ truy cập
    axes = axes.flatten()
    
    # Danh sách các cột dữ liệu và nhãn
    sensor_data_cols = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']
    sensor_labels = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
    y_axis_labels = ['Gia tốc (g)', 'Gia tốc (g)', 'Gia tốc (g)', 'Vận tốc góc (rad/s)', 'Vận tốc góc (rad/s)', 'Vận tốc góc (rad/s)']
    colors = ['red', 'green', 'blue', 'red', 'green', 'blue']
    
    # Vẽ dữ liệu cho mỗi subplot
    for i, col in enumerate(sensor_data_cols):
        axes[i].plot(df['timestamp'], df[col], label=sensor_labels[i], color=colors[i])
        axes[i].set_title(f'{sensor_labels[i]}')
        axes[i].set_ylabel(y_axis_labels[i])
        axes[i].grid(True)
        axes[i].legend(loc='right')
    
    # Calculate common y-axis limits for accelerometer and gyroscope data
    accel_min = df[['accelX', 'accelY', 'accelZ']].values.min()
    accel_max = df[['accelX', 'accelY', 'accelZ']].values.max()
    gyro_min = df[['gyroX', 'gyroY', 'gyroZ']].values.min()
    gyro_max = df[['gyroX', 'gyroY', 'gyroZ']].values.max()

    # Set common y-axis limits for subplots
    for i in range(3):
        axes[i].set_ylim(accel_min - (0.1 * abs(accel_max - accel_min)), accel_max + (0.1 * abs(accel_max - accel_min)))
    for i in range(3, 6):
        axes[i].set_ylim(gyro_min - (0.1 * abs(gyro_max - gyro_min)), gyro_max + (0.1 * abs(gyro_max - gyro_min)))

    # Set xlabel for bottom row subplots (indices 3, 4, 5 in the flattened array)
    for i in range(3, 6):
        axes[i].set_xlabel('Thời gian (ms)')
    
    # Điều chỉnh layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    
    # Lưu đồ thị
    output_dir = Path('sensor_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Tạo thư mục con theo điều kiện đeo
    condition = os.path.basename(file_path).split('_')[0]
    condition_dir = output_dir / condition
    condition_dir.mkdir(exist_ok=True)
    
    # Lưu file với tên mô tả
    filename = os.path.basename(file_path).replace('.csv', '.png')
    plt.savefig(condition_dir / filename)
    plt.close()

def main():
    # Đường dẫn đến thư mục chứa dữ liệu
    data_dir = 'sensor_data'
    
    # Lấy danh sách các file CSV
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ Không tìm thấy file dữ liệu nào trong thư mục sensor_data!")
        return
    
    print("Đang vẽ đồ thị dữ liệu cảm biến...")
    
    # Vẽ đồ thị cho từng file
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        print(f"Đang xử lý file: {file}")
        plot_sensor_data(file_path)
    
    print(f"\n✅ Đã vẽ xong đồ thị cho {len(csv_files)} file!")
    print(f"Đồ thị được lưu trong thư mục: {os.path.abspath('sensor_plots')}")
    print("\nCấu trúc thư mục:")
    print("sensor_plots/")
    print("├── tight/")
    print("│   ├── tight_static_*.png")
    print("│   └── tight_rotation_*.png")
    print("├── loose/")
    print("│   ├── loose_static_*.png")
    print("│   └── loose_rotation_*.png")
    print("└── normal/")
    print("    ├── normal_static_*.png")
    print("    └── normal_rotation_*.png")

if __name__ == "__main__":
    main() 