"""
HANDWRITTEN DIGIT RECOGNIZER (Beginner Friendly)
--------------------------------------------------
This program looks at a handwritten digit image and predicts
which number (0-9) it is, using the MNIST dataset.
"""

# STEP 1: Import the libraries we need
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# STEP 2: Load the MNIST dataset (already built into TensorFlow)
# x = images of digits, y = the actual correct number (label)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training images: {x_train.shape[0]}")
print(f"Testing images: {x_test.shape[0]}")

# STEP 3: Prepare the images
# Pixel values range from 0-255, we scale them to 0-1 (normalize)
# This helps the model learn faster and better
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CNN needs images in 3D shape: (height, width, channels)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# STEP 4: Build the CNN model
# Conv2D layers -> detect patterns like edges and curves
# MaxPooling -> shrinks the image while keeping important info
# Dense layers -> make the final decision on which digit it is
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")   # 10 outputs since digits are 0-9
])

# STEP 5: Compile the model (set training rules)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# STEP 6: Train the model
print("\nTraining the model... this may take a minute or two.\n")
model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1)

# STEP 7: Test the model's accuracy
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# STEP 8: Try predicting on a few new images
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

print("\nSample predictions:")
for i in range(5):
    print(f"Actual digit: {y_test[i]}   -->   Predicted digit: {predicted_labels[i]}")