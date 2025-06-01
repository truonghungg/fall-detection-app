import numpy as np
import pandas as pd
import ruptures as rpt

# Đọc dataset theo chunk để tránh lỗi bộ nhớ
chunk_size = 10000  # Điều chỉnh dựa trên RAM của hệ thống
chunks = pd.read_csv("data/dataset.csv", chunksize=chunk_size)

# Lưu trữ các điểm thay đổi từ tất cả các chunk
all_change_points = []

for chunk in chunks:
    # Lấy cột dữ liệu cảm biến
    signal = chunk[["AccelX", "AccelY", "AccelZ"]].values.astype(np.float32)  # Dùng float32 để giảm bộ nhớ
    
    # Sử dụng thuật toán Pelt để phát hiện điểm thay đổi
    algo = rpt.Pelt(model="rbf").fit(signal)
    change_points = algo.predict(pen=10)  # pen: độ phạt để điều chỉnh số điểm thay đổi
    
    # Điều chỉnh chỉ số điểm thay đổi cho chunk hiện tại
    chunk_start_idx = chunk.index[0]
    adjusted_change_points = [cp + chunk_start_idx for cp in change_points[:-1]]  # Bỏ điểm cuối (kích thước chunk)
    all_change_points.extend(adjusted_change_points)

# In các điểm thay đổi
print("Điểm thay đổi:", all_change_points)

# Phân đoạn dữ liệu dựa trên điểm thay đổi
df = pd.read_csv("dataset.csv", usecols=["Time", "AccelX", "AccelY", "AccelZ", "ActivityLabel"])
segments = []
start = 0
for cp in all_change_points + [len(df)]:
    segment = df.iloc[start:cp]
    segments.append(segment)
    start = cp

# In thông tin các đoạn
for i, segment in enumerate(segments):
    print(f"Đoạn {i+1}: {len(segment)} mẫu, Nhãn phổ biến: {segment['ActivityLabel'].mode()[0]}")