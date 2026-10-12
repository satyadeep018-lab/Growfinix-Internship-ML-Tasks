import os
import tensorflow as tf
from model import build_emotion_cnn

def train_model():
    os.makedirs("models", exist_ok=True)
    model = build_emotion_cnn()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.save("models/emotion_cnn.keras")
    print("Model initialized and saved to 'models/emotion_cnn.keras'.")

if __name__ == "__main__":
    train_model()
