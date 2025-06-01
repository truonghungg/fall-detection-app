import pandas as pd
import numpy as np
from scipy import stats, fft
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import pickle
import os
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def validate_input(df, sensor_cols, window_size, step):
    """Validate input data and parameters"""
    if df.empty:
        raise ValueError("Input DataFrame is empty!")
    if not all(col in df.columns for col in sensor_cols + ['ActivityLabel']):
        raise ValueError("Required columns missing in DataFrame!")
    if window_size <= 0 or step <= 0:
        raise ValueError("Window size and step must be positive!")
    return True

def preprocess_basic(df, sensor_cols):
    """Handle basic preprocessing: drop NA, duplicates, outliers, and smooth data"""
    df = df.dropna().drop_duplicates()
    z_scores = np.abs(stats.zscore(df[sensor_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    for col in sensor_cols:
        df[col] = df[col].rolling(window=3, min_periods=1, center=True).mean()
    return df

def normalize_data(df, sensor_cols, scaler_path='preprocessing/scaler.pkl'):
    """Normalize sensor data and save scaler"""
    scaler = MinMaxScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    os.makedirs('preprocessing', exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    return df

def create_windows(df, sensor_cols, window_size, step):
    """Create windows efficiently using vectorized operations"""
    windows, labels = [], []
    indices = np.arange(0, len(df) - window_size + 1, step)
    for i in indices:
        window = df.iloc[i:i + window_size]
        if len(window['ActivityLabel'].unique()) == 1:
            windows.append(window[sensor_cols].values)
            labels.append(window['ActivityLabel'].iloc[0])
    return np.array(windows), np.array(labels)

def compute_fft_features(X, axis_idx, timesteps):
    """Compute FFT features for a given axis"""
    return np.abs(fft.fft(X[:, :, axis_idx], axis=1))

def add_features(X):
    """Add features with vectorized operations"""
    num_windows, timesteps, num_features = X.shape
    accel_magnitude = np.sqrt(np.sum(X[:, :, :3]**2, axis=2))
    gyro_magnitude = np.sqrt(np.sum(X[:, :, 3:]**2, axis=2))
    fft_accel_x = compute_fft_features(X, 0, timesteps)
    fft_gyro_x = compute_fft_features(X, 3, timesteps)
    freq_range = np.fft.fftfreq(timesteps)
    positive_freq_mask = freq_range >= 0
    fft_accel_mag = np.abs(fft.fft(accel_magnitude, axis=1))[:, positive_freq_mask]
    freq_range = freq_range[positive_freq_mask]
    walking_mask = (freq_range >= 0.5) & (freq_range <= 3.0)
    step_freq = np.zeros((num_windows, timesteps))
    if np.any(walking_mask):
        dominant_freq_idx = np.argmax(fft_accel_mag[:, walking_mask], axis=1)
        step_freq[:] = freq_range[walking_mask][dominant_freq_idx][:, None]
    else:
        dominant_freq_idx = np.argmax(fft_accel_mag, axis=1)
        step_freq[:] = freq_range[dominant_freq_idx][:, None]
    tilt_x = np.arctan2(X[:, :, 0], np.sqrt(X[:, :, 1]**2 + X[:, :, 2]**2))
    tilt_y = np.arctan2(X[:, :, 1], np.sqrt(X[:, :, 0]**2 + X[:, :, 2]**2))
    accel_variance = np.var(X[:, :, :3], axis=2)
    return np.concatenate([
        X,
        accel_magnitude[:, :, np.newaxis],
        gyro_magnitude[:, :, np.newaxis],
        fft_accel_x[:, :, np.newaxis],
        fft_gyro_x[:, :, np.newaxis],
        step_freq[:, :, np.newaxis],
        tilt_x[:, :, np.newaxis],
        tilt_y[:, :, np.newaxis],
        accel_variance[:, :, np.newaxis]
    ], axis=2)

def augment_data(X, y):
    """Chỉ giữ nhiễu, bỏ time warping và đảo ngược thời gian"""
    X_aug = [X]
    y_aug = [y]
    
    # Thêm nhiễu
    noise = np.random.normal(0, 0.02, X.shape)
    X_aug.append(X + noise)
    y_aug.append(y)
    
    return np.vstack(X_aug), np.concatenate(y_aug)

def load_and_preprocess_data(file_path='data/dataset.csv', window_size=100, step=50):
    """Load and preprocess data efficiently"""
    sensor_cols = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist!")
    df = pd.read_csv(file_path)
    validate_input(df, sensor_cols, window_size, step)
    df = preprocess_basic(df, sensor_cols)
    df = normalize_data(df, sensor_cols)
    X, y = create_windows(df, sensor_cols, window_size, step)
    X = add_features(X)
    X_reshaped = X.reshape(X.shape[0], -1)
    ros = RandomOverSampler(random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X_reshaped, y)
    X = X_balanced.reshape(-1, window_size, X.shape[2])
    y = y_balanced
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    with open('preprocessing/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, y_train = augment_data(X_train, y_train)
    return X_train, X_test, y_train, y_test, label_encoder

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()
    os.makedirs('preprocessing', exist_ok=True)
    with open('preprocessing/processed_data.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test, label_encoder), f)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Classes: {label_encoder.classes_}")