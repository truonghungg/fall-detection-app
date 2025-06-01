import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


with open('preprocessing/processed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test, label_encoder = pickle.load(f)
    print(f"✅ Loaded data successfully!")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    print(f"Classes: {label_encoder.classes_}")


window_size = 100
num_features = 14
if X_train.shape[1:] != (window_size, num_features):
    print(f"❌ X_train shape {X_train.shape[1:]} does not match expected ({window_size}, {num_features})")
    exit(1)
    
num_timesteps = X_train.shape[1]
num_features = X_train.shape[2]
num_classes = len(label_encoder.classes_)
print(f"Timesteps: {num_timesteps}, Features: {num_features}, Classes: {num_classes}")


def build_cnn_lstm_model():
    model = Sequential([
        Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(num_timesteps, num_features)),
        Dropout(0.3),
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        Dropout(0.3),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn_lstm_model()
model.summary()

# Train model
batch_size = 32 
epochs = 50  

print(f"🚀 Training model with {epochs} epochs, batch size {batch_size}...")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    verbose=1,
)
final_model_path = "models/cnn_lstm.h5"
model.save(final_model_path)
# Save training history
with open('models/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("✅ Saved training history")

# Evaluate model
print("📊 Evaluating model...")
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('models/confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('models/training_history.png')
plt.close()

print("✅ Saved confusion matrix and training history plots")