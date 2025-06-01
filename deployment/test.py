import asyncio
import websockets
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from aiohttp import web
import os
import time
from collections import deque
import pickle
import aiohttp_cors
from sklearn.metrics import accuracy_score, classification_report
import ssl
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print(f"TensorFlow version: {tf.__version__}")

# Configuration
model_path = 'models/cnn_lstm.h5'
scaler_path = 'scaler.pkl'
data_path = 'preprocessing/processed_data_fixed.pkl'
window_size = 100
pi_uri = os.getenv('PI_URI', "ws://192.168.100.86:8080")
prediction_interval = 0.2
clients = set()

# Email Configuration
EMAIL_SENDER = os.getenv('EMAIL_SENDER', 'hung0108az@gmail.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'itkp uaxn afni kdzi')
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT', 'hung0108a@gmail.com')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))

# Fall detection configuration
fall_buffer = deque(maxlen=5)  # Lưu 5 dự đoán gần nhất
fall_threshold = 0.9  # Tăng ngưỡng tin cậy lên 0.9
email_cooldown = 300  
last_email_time = 0

def send_email_alert():
    """Gửi email cảnh báo khi phát hiện ngã"""
    global last_email_time
    current_time = time.time()
    
    if current_time - last_email_time < email_cooldown:
        print("⚠️ Email cooldown active, skipping alert")
        return
    
    try:
        # Tạo message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECIPIENT
        msg['Subject'] = "⚠️ CẢNH BÁO: Phát hiện ngã!"
        
        body = """
        <html>
        <body>
            <h2 style="color: red;">⚠️ CẢNH BÁO: Phát hiện ngã!</h2>
            <p>Hệ thống đã phát hiện hành động ngã với độ tin cậy cao.</p>
            <p>Vui lòng kiểm tra ngay lập tức!</p>
            <p>Thời gian phát hiện: {}</p>
        </body>
        </html>
        """.format(time.strftime("%Y-%m-%d %H:%M:%S"))
        
        msg.attach(MIMEText(body, 'html'))
        
        # Kết nối và gửi email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print("✅ Email alert sent successfully")
        last_email_time = current_time
        
    except Exception as e:
        print(f"❌ Error sending email: {e}")

# Load model
try:
    model = load_model(model_path)
    print("✅ Loaded CNN-LSTM model")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Load scaler
try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Loaded scaler")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    exit(1)

# Load test data and label encoder
try:
    with open(data_path, 'rb') as f:
        _, _, _, _, label_encoder = pickle.load(f)
    action_labels = list(label_encoder.classes_)
    print(f"✅ Loaded data, classes: {action_labels}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Feature extraction
def extract_features(data):
    window_size = data.shape[0]
    
    # Basic features
    accel_magnitude = np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)
    gyro_magnitude = np.sqrt(data[:, 3]**2 + data[:, 4]**2 + data[:, 5]**2)
    
    # FFT features
    fft_accel_x = np.abs(np.fft.fft(data[:, 0]))
    fft_gyro_x = np.abs(np.fft.fft(data[:, 3]))
    
    # Step frequency
    fft_accel_mag = np.abs(np.fft.fft(accel_magnitude))
    freq_range = np.fft.fftfreq(window_size)
    positive_freq_mask = freq_range >= 0
    freq_range = freq_range[positive_freq_mask]
    fft_accel_mag = fft_accel_mag[positive_freq_mask]
    
    walking_mask = (freq_range >= 0.5) & (freq_range <= 3.0)
    if np.any(walking_mask):
        dominant_freq_idx = np.argmax(fft_accel_mag[walking_mask])
        step_freq = np.full(window_size, freq_range[walking_mask][dominant_freq_idx])
    else:
        dominant_freq_idx = np.argmax(fft_accel_mag)
        step_freq = np.full(window_size, freq_range[dominant_freq_idx])
    
    # Tilt angles
    tilt_x = np.arctan2(data[:, 0], np.sqrt(data[:, 1]**2 + data[:, 2]**2))
    tilt_y = np.arctan2(data[:, 1], np.sqrt(data[:, 0]**2 + data[:, 2]**2))
    
    # Combine features
    return np.concatenate([
        data,                          # Original 6 features
        accel_magnitude[:, np.newaxis],    # Magnitude of acceleration
        gyro_magnitude[:, np.newaxis],     # Magnitude of gyroscope
        fft_accel_x[:, np.newaxis],        # FFT of AccelX
        fft_gyro_x[:, np.newaxis],         # FFT of GyroX
        step_freq[:, np.newaxis],          # Step frequency
        tilt_x[:, np.newaxis],             # Tilt angle X
        tilt_y[:, np.newaxis],             # Tilt angle Y
    ], axis=1)

# Predict action
async def predict(buffer):
    try:
        input_data = np.array(buffer, dtype=np.float32)
        if input_data.shape != (window_size, 6):
            return None, None
        
        input_data_scaled = scaler.transform(input_data)
        input_data_features = extract_features(input_data_scaled)
        input_batch = np.expand_dims(input_data_features, axis=0)
        
        pred = model(input_batch, training=False)[0]
        predicted_class = np.argmax(pred)
        confidence = float(pred[predicted_class])
        predicted_action = action_labels[predicted_class]
        
        # Fall detection
        if predicted_action == "fall" and confidence > fall_threshold:
            fall_buffer.append(True)
            if len(fall_buffer) == 5 and all(fall_buffer):
                print("⚠️ Fall detected!")
                send_email_alert()
        else:
            fall_buffer.append(False)
        
        return predicted_action, confidence
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None

# WebSocket handlers
async def handle_client(websocket, path=None):
    clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except websockets.ConnectionClosed:
        pass
    finally:
        clients.remove(websocket)

async def receive_data():
    buffer = deque(maxlen=window_size)
    last_prediction_time = 0
    
    while True:
        try:
            async with websockets.connect(pi_uri) as pi_socket:
                print("✅ Connected to Raspberry Pi")
                while True:
                    batch = json.loads(await pi_socket.recv())
                    current_time = time.time()
                    
                    for sensor_data in batch:
                        values = [sensor_data.get(key, 0.0) for key in ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"]]
                        buffer.append(values)
                        
                        if len(buffer) == window_size and (current_time - last_prediction_time) >= prediction_interval:
                            last_prediction_time = current_time
                            action, confidence = await predict(list(buffer))
                            if action:
                                message = {
                                    **sensor_data,
                                    "action": action,
                                    "confidence": confidence
                                }
                                await broadcast_result(message)
        except Exception as e:
            print(f"❌ Pi connection error: {e}")
            await asyncio.sleep(2)

async def broadcast_result(message):
    for client in list(clients):
        try:
            await client.send(json.dumps(message))
        except:
            clients.remove(client)

async def serve_html(request):
    file_path = os.path.join(os.getcwd(), "deployment/index.html")
    if os.path.exists(file_path):
        return web.FileResponse(file_path)
    return web.Response(text="File not found", status=404)

# WSGI Application for Render
app = web.Application()
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
        allow_methods="*",
    )
})

app.router.add_get('/', serve_html)
for route in list(app.router.routes()):
    cors.add(route)

# Create WSGI application
application = app

if __name__ == "__main__":
    try:
        port = int(os.getenv('PORT', 8080))
        web.run_app(app, host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        print("⏹ Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")