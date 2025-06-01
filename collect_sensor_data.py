import numpy as np
import pandas as pd
import time
from datetime import datetime
import os
import math

class SensorDataCollector:
    def __init__(self):
        self.data_dir = "sensor_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Các trạng thái đeo
        self.wearing_conditions = [
            "tight",  # Đeo chặt
            "loose",  # Đeo lỏng
            "normal"  # Đeo bình thường
        ]
        
        # Các loại chuyển động
        self.movements = [
            "static",  # Giữ nguyên
            "rotation"  # Xoay cánh tay từ trái sang phải
        ]
        
        # Tên các cảm biến
        self.sensors = [
            "accelerometer",
            "gyroscope",
        ]

        # Hệ số chuyển đổi
        self.GRAVITY = 9.80665  # m/s²
        self.DEG_TO_RAD = math.pi / 180.0

    def get_sensor_data(self, condition, movement):
        """
        Tạo dữ liệu mô phỏng dựa trên điều kiện đeo và chuyển động
        """
        # Tạo nhiễu ngẫu nhiên
        noise = np.random.normal(0, 0.1, 6)
        
        if movement == "static":
            # Dữ liệu khi giữ nguyên
            accel = [0, 0, self.GRAVITY]  # Trọng trường hướng xuống
            gyro = [0, 0, 0]  # Không có chuyển động quay
        else:  # rotation
            # Dữ liệu khi xoay cánh tay
            t = time.time()
            accel = [
                np.sin(t) * 2,  # Dao động theo trục X
                np.cos(t) * 2,  # Dao động theo trục Y
                self.GRAVITY + np.sin(t) * 0.5  # Trọng trường + dao động nhỏ
            ]
            gyro = [
                np.cos(t) * 2,  # Vận tốc góc theo trục X
                np.sin(t) * 2,  # Vận tốc góc theo trục Y
                0  # Ít xoay theo trục Z
            ]
        
        # Thêm nhiễu dựa trên điều kiện đeo
        if condition == "loose":
            noise *= 2  # Nhiễu lớn hơn khi đeo lỏng
        elif condition == "tight":
            noise *= 0.5  # Nhiễu nhỏ hơn khi đeo chặt
        
        # Áp dụng nhiễu
        accel = [a + n for a, n in zip(accel, noise[:3])]
        gyro = [g + n for g, n in zip(gyro, noise[3:])]
        
        return {
            "accelerometer": {"x": round(accel[0], 6), "y": round(accel[1], 6), "z": round(accel[2], 6)},
            "gyroscope": {"x": round(gyro[0], 6), "y": round(gyro[1], 6), "z": round(gyro[2], 6)}
        }

    def collect_data(self, duration=10, sample_rate=100):
        """
        Thu thập dữ liệu từ cảm biến
        duration: thời gian thu thập (giây)
        sample_rate: tần số lấy mẫu (Hz)
        """
        print("\n=== BẮT ĐẦU THU THẬP DỮ LIỆU ===")
        
        for condition in self.wearing_conditions:
            print(f"\nTrạng thái đeo: {condition}")
            
            for movement in self.movements:
                print(f"\nChuyển động: {movement}")
                print("Chuẩn bị trong 3 giây...")
                time.sleep(3)
                
                # Tạo timestamp cho file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.data_dir}/{condition}_{movement}_{timestamp}.csv"
                
                # Số mẫu cần thu thập
                num_samples = duration * sample_rate
                
                # Tạo DataFrame để lưu dữ liệu
                data = {
                    'timestamp': [],
                    'sensor': [],
                    'x': [],
                    'y': [],
                    'z': [],
                    'value': []
                }
                
                print(f"Đang thu thập dữ liệu trong {duration} giây...")
                start_time = time.time()
                
                for i in range(num_samples):
                    current_time = time.time() - start_time
                    
                    # Đọc dữ liệu từ cảm biến
                    sensor_data = self.get_sensor_data(condition, movement)
                    
                    # Lưu dữ liệu gia tốc kế
                    data['timestamp'].append(current_time)
                    data['sensor'].append('accelerometer')
                    data['x'].append(sensor_data['accelerometer']['x'])
                    data['y'].append(sensor_data['accelerometer']['y'])
                    data['z'].append(sensor_data['accelerometer']['z'])
                    data['value'].append(None)
                    
                    # Lưu dữ liệu con quay hồi chuyển
                    data['timestamp'].append(current_time)
                    data['sensor'].append('gyroscope')
                    data['x'].append(sensor_data['gyroscope']['x'])
                    data['y'].append(sensor_data['gyroscope']['y'])
                    data['z'].append(sensor_data['gyroscope']['z'])
                    data['value'].append(None)
                    
                    # Đợi đến mẫu tiếp theo
                    time.sleep(1/sample_rate)
                
                # Lưu dữ liệu vào file CSV
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                print(f"Đã lưu dữ liệu vào file: {filename}")
                
                print("Nghỉ 2 giây trước khi tiếp tục...")
                time.sleep(2)

def main():
    collector = SensorDataCollector()
    
    print("Chương trình thu thập dữ liệu cảm biến")
    print("Các trạng thái đeo:", collector.wearing_conditions)
    print("Các loại chuyển động:", collector.movements)
    print("Các loại cảm biến:", collector.sensors)
    
    input("\nNhấn Enter để bắt đầu thu thập dữ liệu...")
    
    collector.collect_data(duration=10, sample_rate=100)
    
    print("\n=== HOÀN THÀNH THU THẬP DỮ LIỆU ===")

if __name__ == "__main__":
    main() 