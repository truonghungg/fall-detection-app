from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import json
import time
from collections import deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
from firebase_config import save_prediction, save_fall_alert

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='../frontend/build')
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
model_path = 'models/cnn_lstm.h5'
scaler_path = 'scaler.pkl'
data_path = 'preprocessing/processed_data_fixed.pkl'
window_size = 100
prediction_interval = 0.2

# Email Configuration
EMAIL_SENDER = os.getenv('EMAIL_SENDER', 'hung0108az@gmail.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'itkp uaxn afni kdzi')
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT', 'hung0108a@gmail.com')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))

# Fall detection configuration
fall_buffer = deque(maxlen=5)
fall_threshold = 0.9
email_cooldown = 300
last_email_time = 0

def send_email_alert():
    global last_email_time
    current_time = time.time()
    
    if current_time - last_email_time < email_cooldown:
        print("⚠️ Email cooldown active, skipping alert")
        return
    
    try:
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
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print("✅ Email alert sent successfully")
        last_email_time = current_time
        
    except Exception as e:
        print(f"❌ Error sending email: {e}")

# Load model and data
try:
    model = load_model(model_path)
    print("✅ Loaded CNN-LSTM model")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Loaded scaler")
    
    with open(data_path, 'rb') as f:
        _, _, _, _, label_encoder = pickle.load(f)
    action_labels = list(label_encoder.classes_)
    print(f"✅ Loaded data, classes: {action_labels}")
except Exception as e:
    print(f"❌ Error loading model/data: {e}")
    exit(1)

def extract_features(data):
    window_size = data.shape[0]
    
    accel_magnitude = np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)
    gyro_magnitude = np.sqrt(data[:, 3]**2 + data[:, 4]**2 + data[:, 5]**2)
    
    fft_accel_x = np.abs(np.fft.fft(data[:, 0]))
    fft_gyro_x = np.abs(np.fft.fft(data[:, 3]))
    
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
    
    tilt_x = np.arctan2(data[:, 0], np.sqrt(data[:, 1]**2 + data[:, 2]**2))
    tilt_y = np.arctan2(data[:, 1], np.sqrt(data[:, 0]**2 + data[:, 2]**2))
    
    return np.concatenate([
        data,
        accel_magnitude[:, np.newaxis],
        gyro_magnitude[:, np.newaxis],
        fft_accel_x[:, np.newaxis],
        fft_gyro_x[:, np.newaxis],
        step_freq[:, np.newaxis],
        tilt_x[:, np.newaxis],
        tilt_y[:, np.newaxis],
    ], axis=1)

def predict(buffer):
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

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('sensor_data')
def handle_sensor_data(data):
    try:
        values = [data.get(key, 0.0) for key in ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"]]
        action, confidence = predict(values)
        if action:
            result = {
                **data,
                "action": action,
                "confidence": confidence
            }
            # Save prediction to Firebase
            save_prediction(result)
            
            # If it's a fall, save to fall_alerts collection
            if action == "fall" and confidence > fall_threshold:
                save_fall_alert(result)
            
            socketio.emit('prediction_result', result)
    except Exception as e:
        print(f"❌ Error processing sensor data: {e}")

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    socketio.run(app, host='0.0.0.0', port=port) 