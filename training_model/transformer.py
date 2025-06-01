import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

print(f"TensorFlow version: {tf.__version__}")

# Tạo thư mục models
os.makedirs('models', exist_ok=True)

# Tải dữ liệu
print("Đang tải dữ liệu...")
with open('preprocessing/processed_data_with_transitions.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test, label_encoder = pickle.load(f)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Classes: {label_encoder.classes_}")

# Thông số mô hình
timesteps = X_train.shape[1]
features = X_train.shape[2]
num_classes = len(label_encoder.classes_)

# Transformer block cho TF 2.18.0
def transformer_encoder(inputs, key_dim, num_heads, ff_dim, dropout=0):
    # Self-attention layer
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(inputs, inputs)
    attention_output = layers.Dropout(dropout)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward layer
    ffn_output = layers.Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    
    return layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)

# Xây dựng mô hình Transformer
def build_model():
    inputs = keras.Input(shape=(timesteps, features))
    
    # Positional encoding đơn giản
    positions = np.arange(timesteps) / timesteps
    position_encoding = np.zeros((1, timesteps, features))
    for i in range(features):
        position_encoding[0, :, i] = positions
    
    x = inputs + tf.constant(position_encoding, dtype=tf.float32)
    
    # Transformer blocks
    x = transformer_encoder(x, key_dim=32, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, key_dim=32, num_heads=4, ff_dim=128, dropout=0.1)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

# Xây dựng mô hình
print("Đang xây dựng mô hình Transformer...")
model = build_model()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Tóm tắt mô hình
model.summary()

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(
        'models/transformer_tf218.h5', save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)
]

# Huấn luyện mô hình
print("Đang huấn luyện mô hình Transformer...")
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Đánh giá mô hình
print("Đánh giá mô hình...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Dự đoán
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)

# Lưu mô hình
model.save('models/transformer.h5', save_format='h5')
print("✅ Đã lưu mô hình tại models/transformer.h5")

# Lưu config mô hình để dễ đọc lại
with open('models/transformer_config.pkl', 'wb') as f:
    model_config = {
        'timesteps': timesteps,
        'features': features,
        'num_classes': num_classes
    }
    pickle.dump(model_config, f)
print("✅ Đã lưu cấu hình mô hình")

# Lưu ngưỡng lớp
class_thresholds = {cls: 0.5 for cls in label_encoder.classes_}
for cls in label_encoder.classes_:
    if 'fall' in cls.lower():
        class_thresholds[cls] = 0.9

with open('models/class_thresholds.pkl', 'wb') as f:
    pickle.dump(class_thresholds, f)
print("✅ Đã lưu ngưỡng lớp")

# Vẽ biểu đồ huấn luyện
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('models/transformer_tf218_history.png')
print("✅ Đã lưu biểu đồ huấn luyện")

print("Huấn luyện hoàn tất!")