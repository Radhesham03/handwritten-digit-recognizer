# Handwritten Digit Recognizer

A beginner-friendly deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the famous MNIST dataset.

## 📌 Overview

This project demonstrates how a CNN can learn to "see" and classify handwritten digit images. The model is trained on thousands of example digit images and learns to recognize patterns like curves and edges that make up each number.

## 🛠️ Tech Stack

- **Python 3**
- **TensorFlow / Keras** — for building and training the CNN
- **NumPy** — for numerical operations on image data

## ⚙️ How It Works

1. The MNIST dataset (70,000 handwritten digit images) is loaded directly from Keras.
2. Images are normalized (pixel values scaled between 0 and 1) for better training.
3. A CNN is built using:
   - `Conv2D` layers — detect visual patterns (edges, curves)
   - `MaxPooling2D` layers — reduce image size while keeping key features
   - `Dense` layers — make the final classification decision
4. The model is trained on 60,000 training images.
5. The model's accuracy is evaluated on 10,000 unseen test images.
6. The trained model predicts digits for new sample images.

## 🚀 How to Run

1. Make sure Python is installed (this project was built and tested using Python 3.12).
2. Install the required libraries:
   ```bash
   pip install tensorflow numpy
   ```
3. Run the script:
   ```bash
   python digit_recognizer.py
   ```

## 📊 Sample Output

```
Test Accuracy: 98.81%

Sample predictions:
Actual digit: 7   -->   Predicted digit: 7
Actual digit: 2   -->   Predicted digit: 2
Actual digit: 1   -->   Predicted digit: 1
Actual digit: 0   -->   Predicted digit: 0
Actual digit: 4   -->   Predicted digit: 4
```

## 🔮 Future Improvements

- Add a simple drawing canvas (web/GUI) so users can draw their own digit and get a live prediction
- Train for more epochs to push accuracy even higher
- Experiment with deeper CNN architectures or data augmentation
- Visualize misclassified digits to understand model weaknesses

## 📁 Project Files

- `digit_recognizer.py` — main Python script containing the full implementation

---

*This project was built as part of a machine learning training assignment.*
