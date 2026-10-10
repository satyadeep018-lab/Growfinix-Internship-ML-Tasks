import tensorflow as tf
from tensorflow.keras import layers, models

def build_emotion_cnn(input_shape=(128, 130, 1), num_classes=3):
    """
    Builds a compact 2D Convolutional Neural Network for Log-Mel Spectrogram classification.
    
    Architecture:
      - 3 Convolutional Blocks with Batch Normalization, ReLU, and MaxPooling
      - Spatial Dropout to prevent overfitting
      - Global Average Pooling to reduce parameter count and keep file size under GitHub limits
      - Fully connected classification head with Softmax
    """
    inputs = layers.Input(shape=input_shape, name="spectrogram_input")

    # Block 1: Low-level acoustic edges & harmonic ridges
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # Block 2: Mid-level formant transitions & pitch contours
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3: High-level emotional timbre patterns
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.3, name="drop3")(x)

    # Global Average Pooling eliminates massive dense layer weights, keeping model < 8MB
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # Dense classification head
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="drop_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SpeechEmotionCNN")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

if __name__ == "__main__":
    net = build_emotion_cnn()
    net.summary()
