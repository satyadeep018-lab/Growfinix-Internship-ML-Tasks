import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_emotion_cnn

def train_emotion_model(
    processed_dir="data/processed",
    model_save_path="models/emotion_cnn.keras",
    history_plot_path="outputs/training_history.png",
    epochs=25,
    batch_size=16
):
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load preprocessed feature tensors
    X_train_path = os.path.join(processed_dir, "X_train.npy")
    y_train_path = os.path.join(processed_dir, "y_train.npy")
    X_val_path = os.path.join(processed_dir, "X_val.npy")
    y_val_path = os.path.join(processed_dir, "y_val.npy")

    if not os.path.exists(X_train_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {processed_dir}. "
            "Please run 'python prepare_data.py --generate-samples' first."
        )

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)

    print(f"[Train] Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    input_shape = X_train.shape[1:]

    # 2. Instantiate CNN model
    model = build_emotion_cnn(input_shape=input_shape, num_classes=3)
    model.summary()

    # 3. Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_save_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    # 4. Train
    print("[Train] Starting CNN model training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Save final model if not saved by checkpoint
    model.save(model_save_path)
    file_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
    print(f"[Train] Model saved successfully to '{model_save_path}' ({file_size_mb:.2f} MB).")

    # 6. Plot & save training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history.history["loss"], label="Train Loss", color="#0284c7")
    axes[0].plot(history.history["val_loss"], label="Val Loss", color="#ea580c")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Accuracy plot
    axes[1].plot(history.history["accuracy"], label="Train Accuracy", color="#0284c7")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy", color="#16a34a")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(history_plot_path, dpi=200)
    plt.close()
    print(f"[Train] Training curves saved to '{history_plot_path}'.")

if __name__ == "__main__":
    train_emotion_model()
