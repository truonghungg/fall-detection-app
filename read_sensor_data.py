import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime
import os
import time

class SensorDataReader:
    def __init__(self):
        self.data_dir = "sensor_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Các trạng thái đeo
        self.wearing_conditions = {
            "1": "tight",    # Đeo chặt
            "2": "loose",    # Đeo lỏng
            "3": "normal"    # Đeo bình thường
        }
        
        # Các loại chuyển động
        self.movements = {
            "1": "static",   # Giữ nguyên
            "2": "rotation"  # Xoay cánh tay từ trái sang phải
        }
        
        # Cấu hình thu thập dữ liệu
        self.sampling_rate = 50  # Hz
        self.duration = 15       # seconds
        self.expected_samples = self.sampling_rate * self.duration  # 750 samples
        self.sample_interval = 1.0 / self.sampling_rate  # 0.02 seconds between samples

    def select_conditions(self):
        """Chọn trạng thái đeo và chuyển động"""
        print("\n=== CHỌN TRẠNG THÁI ĐEO ===")
        print("1. Đeo chặt")
        print("2. Đeo lỏng")
        print("3. Đeo bình thường")
        
        while True:
            choice = input("\nChọn trạng thái đeo (1-3): ").strip()
            if choice in self.wearing_conditions:
                selected_condition = self.wearing_conditions[choice]
                break
            print("❌ Lựa chọn không hợp lệ! Vui lòng chọn lại.")

        print("\n=== CHỌN CHUYỂN ĐỘNG ===")
        print("1. Giữ nguyên")
        print("2. Xoay cánh tay từ trái sang phải")
        
        while True:
            choice = input("\nChọn chuyển động (1-2): ").strip()
            if choice in self.movements:
                selected_movement = self.movements[choice]
                break
            print("❌ Lựa chọn không hợp lệ! Vui lòng chọn lại.")

        return selected_condition, selected_movement

    async def collect_data(self, websocket, condition, movement):
        """
        Thu thập dữ liệu từ WebSocket server với tần số 50Hz trong 15 giây
        websocket: Kết nối WebSocket
        condition: Trạng thái đeo
        movement: Loại chuyển động
        """
        print(f"\nĐang thu thập dữ liệu cho {condition} - {movement}")
        print(f"Tần số lấy mẫu: {self.sampling_rate}Hz")
        print(f"Thời gian thu thập: {self.duration} giây")
        print(f"Số mẫu dự kiến: {self.expected_samples}")
        print("Chuẩn bị trong 3 giây...")
        await asyncio.sleep(3)

        # Tạo timestamp cho file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_dir}/{condition}_{movement}_{timestamp}.csv"

        # Tạo DataFrame để lưu dữ liệu
        data = {
            'timestamp': [],
            'accelX': [],
            'accelY': [],
            'accelZ': [],
            'gyroX': [],
            'gyroY': [],
            'gyroZ': []
        }

        print(f"Đang thu thập dữ liệu trong {self.duration} giây...")
        start_time = time.time()
        end_time = start_time + self.duration
        last_data_time = start_time
        samples_collected = 0
        next_sample_time = start_time

        # Tạo buffer để lưu dữ liệu tạm thời
        data_buffer = []

        while time.time() < end_time and samples_collected < self.expected_samples:
            try:
                # Đặt timeout ngắn hơn để nhận dữ liệu nhanh hơn
                async with asyncio.timeout(0.001):  # 1ms timeout
                    message = await websocket.recv()
                    current_time = time.time()
                    last_data_time = current_time

                    # Thêm dữ liệu vào buffer
                    batch_data = json.loads(message)
                    data_buffer.extend(batch_data)

                    # Xử lý dữ liệu từ buffer
                    while data_buffer and current_time >= next_sample_time and samples_collected < self.expected_samples:
                        sensor_data = data_buffer.pop(0)  # Lấy mẫu đầu tiên từ buffer
                        
                        data['timestamp'].append(sensor_data.get('timestamp', time.time() * 1000))
                        data['accelX'].append(sensor_data.get('accelX', 0.0))
                        data['accelY'].append(sensor_data.get('accelY', 0.0))
                        data['accelZ'].append(sensor_data.get('accelZ', 0.0))
                        data['gyroX'].append(sensor_data.get('gyroX', 0.0))
                        data['gyroY'].append(sensor_data.get('gyroY', 0.0))
                        data['gyroZ'].append(sensor_data.get('gyroZ', 0.0))
                        
                        samples_collected += 1
                        next_sample_time = start_time + (samples_collected * self.sample_interval)

                        # Hiển thị tiến trình
                        if samples_collected % 50 == 0:  # Hiển thị mỗi giây
                            elapsed_time = current_time - start_time
                            remaining_time = self.duration - elapsed_time
                            print(f"Đã thu thập {samples_collected}/{self.expected_samples} mẫu "
                                  f"({samples_collected/self.expected_samples*100:.1f}%) "
                                  f"- Thời gian: {elapsed_time:.1f}s "
                                  f"- Còn lại: {remaining_time:.1f}s")

            except asyncio.TimeoutError:
                # Kiểm tra nếu không nhận được dữ liệu trong 2 giây
                if time.time() - last_data_time > 2.0:
                    print("❌ Không nhận được dữ liệu trong 2 giây, thử kết nối lại...")
                    return False
                continue
            except websockets.exceptions.ConnectionClosed:
                print("❌ Mất kết nối với server!")
                return False
            except Exception as e:
                print(f"❌ Lỗi khi nhận dữ liệu: {e}")
                return False

        # Tính toán thống kê về tần số lấy mẫu
        elapsed_time = time.time() - start_time
        actual_rate = samples_collected / elapsed_time
        print(f"\nThống kê thu thập dữ liệu:")
        print(f"- Số mẫu đã thu thập: {samples_collected}")
        print(f"- Thời gian thực tế: {elapsed_time:.2f} giây")
        print(f"- Tần số lấy mẫu thực tế: {actual_rate:.2f} Hz")

        # Lưu dữ liệu vào file CSV
        if len(data['timestamp']) > 0:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"✅ Đã lưu dữ liệu vào file: {filename}")
            return True
        else:
            print("❌ Không có dữ liệu để lưu!")
            return False

    async def try_connect(self, server_url, max_retries=3):
        """Thử kết nối đến server với số lần thử lại"""
        for attempt in range(max_retries):
            try:
                print(f"\nĐang kết nối đến server... (Lần thử {attempt + 1}/{max_retries})")
                async with asyncio.timeout(10):  # Set timeout for the entire connection attempt
                    websocket = await websockets.connect(server_url)
                    print("✅ Đã kết nối với server!")
                    return websocket
            except asyncio.TimeoutError:
                print("❌ Kết nối timeout! Kiểm tra lại kết nối mạng.")
            except ConnectionRefusedError:
                print("❌ Không thể kết nối đến server! Kiểm tra lại địa chỉ IP và port.")
            except websockets.exceptions.InvalidStatusCode as e:
                print(f"❌ Lỗi kết nối: {e}")
            except Exception as e:
                print(f"❌ Lỗi không xác định: {e}")
            
            if attempt < max_retries - 1:
                print("Đang thử kết nối lại sau 3 giây...")
                await asyncio.sleep(3)
        
        return None

    async def start_collection(self, server_url="ws://192.168.100.86:8080"):
        """
        Bắt đầu thu thập dữ liệu
        server_url: URL của WebSocket server
        """
        while True:
            websocket = await self.try_connect(server_url)
            if not websocket:
                print("❌ Không thể kết nối đến server sau nhiều lần thử!")
                retry = input("Bạn có muốn thử kết nối lại không? (y/n): ").strip().lower()
                if retry != 'y':
                    return
                continue

            try:
                while True:
                    # Chọn trạng thái đeo và chuyển động
                    condition, movement = self.select_conditions()
                    
                    # Thu thập dữ liệu
                    success = await self.collect_data(websocket, condition, movement)
                    
                    if not success:
                        print("❌ Lỗi khi thu thập dữ liệu, thử kết nối lại...")
                        break
                    
                    # Hỏi người dùng có muốn tiếp tục không
                    choice = input("\nBạn có muốn thu thập thêm dữ liệu không? (y/n): ").strip().lower()
                    if choice != 'y':
                        return

            except websockets.exceptions.ConnectionClosed:
                print("❌ Mất kết nối với server!")
            except Exception as e:
                print(f"❌ Lỗi: {e}")
            finally:
                await websocket.close()

async def main():
    reader = SensorDataReader()
    
    print("Chương trình thu thập dữ liệu cảm biến")
    print(f"Tần số lấy mẫu: {reader.sampling_rate}Hz")
    print(f"Thời gian thu thập: {reader.duration} giây")
    print(f"Số mẫu mỗi lần thu thập: {reader.expected_samples}")
    
    # Nhập địa chỉ IP của Raspberry Pi
    while True:
        server_ip = input("\nNhập địa chỉ IP của Raspberry Pi (mặc định: 192.168.100.86): ").strip()
        if not server_ip:
            server_ip = "192.168.100.86"
        
        # Kiểm tra định dạng IP
        if all(part.isdigit() and 0 <= int(part) <= 255 for part in server_ip.split('.')):
            break
        print("❌ Địa chỉ IP không hợp lệ! Vui lòng nhập lại.")
    
    server_url = f"ws://{server_ip}:8080"
    
    input("\nNhấn Enter để bắt đầu thu thập dữ liệu...")
    
    await reader.start_collection(server_url)
    
    print("\n=== HOÀN THÀNH THU THẬP DỮ LIỆU ===")

if __name__ == "__main__":
    asyncio.run(main()) 