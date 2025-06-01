import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
import pickle
import os

# --- Phần 1: Tải dữ liệu đã tiền xử lý ---
processed_data_path = 'preprocessing/processed_data.pkl'

if not os.path.exists(processed_data_path):
    print(f"Lỗi: Không tìm thấy file dữ liệu đã tiền xử lý tại {processed_data_path}.")
    print("Vui lòng chạy script tiền xử lý (preprocessing/test.py) trước.")
    exit()
else:
    with open(processed_data_path, 'rb') as f:
        X_train, X_test, y_train, y_test, label_encoder = pickle.load(f)

    print(f"Đã tải dữ liệu thành công.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Classes: {label_encoder.classes_}")

    # Số lượng lớp (hoạt động)
    num_classes = len(label_encoder.classes_)
    # Kích thước input cho mô hình (số bước thời gian, số đặc trưng)
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Chuyển đổi nhãn sang dạng one-hot encoding (nếu script tiền xử lý chưa làm)
    if len(y_train.shape) == 1 or y_train.shape[1] != num_classes:
         y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
         y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    else:
         y_train_one_hot = y_train
         y_test_one_hot = y_test

    print(f"y_train after one-hot encoding shape: {y_train_one_hot.shape}")
    print(f"y_test after one-hot encoding shape: {y_test_one_hot.shape}")

    # --- Phần 2: Xây dựng và Huấn luyện mô hình CNN ---
    def build_cnn_model(input_shape, num_classes):
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    print("\nXây dựng mô hình CNN...")
    cnn_model = build_cnn_model(input_shape, num_classes)
    cnn_model.summary()

    print("\nHuấn luyện mô hình CNN...")
    cnn_history = cnn_model.fit(X_train, y_train_one_hot, epochs=50, batch_size=32, validation_split=0.2)

    # --- Phần 3: Đánh giá mô hình CNN ---
    print("\nĐánh giá mô hình CNN...")
    cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test_one_hot, verbose=0)
    print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")
    
    # Lưu mô hình CNN (tùy chọn)
    # cnn_model.save('training_model/cnn_model.h5')
    # print("Mô hình CNN đã được lưu tại training_model/cnn_model.h5") 