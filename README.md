# 👕 Fashion MNIST Classification: Neural Network with TensorFlow & Keras

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Model-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-brightgreen.svg)](https://matplotlib.org/)

This project implements an **Artificial Neural Network (ANN)** using Python, **TensorFlow**, and **Keras** to classify clothing images from the classic **Fashion MNIST** dataset. The primary objective is to build, train, and evaluate a sequential deep learning model capable of predicting apparel categories based on pixel features.

---

## 📂 Project Structure & Workflow

The script (`tensor.py`) performs the following standard deep learning workflow:
1. **Dataset Loading:** Imports and splits the `Fashion MNIST` dataset via `keras.datasets.fashion_mnist` into training and testing sets.
2. **Data Normalization:** Scales pixel values from `[0, 255]` to a normalized range of `[0.0, 1.0]` by dividing by `255.0` to stabilize gradient descent.
3. **Model Architecture:** Constructs a Sequential neural network containing:
   - A **Flatten** layer to convert 28x28 pixel matrices into a 1D vector.
   - A **Dense hidden layer** with 50 neurons and `ReLU` activation.
   - A **Dense output layer** with 10 neurons and `Softmax` activation for multi-class probability distribution.
4. **Model Compilation:** Configures the model using the `adam` optimizer, `sparse_categorical_crossentropy` loss function, and `accuracy` metrics.
5. **Model Training & Evaluation:** Trains the network over multiple epochs and evaluates generalization performance on test data (`model.evaluate`).
6. **Performance Visualization:** Generates side-by-side `matplotlib` graphs tracking training accuracy and loss evolution across epochs.

---

## ⚙️ Data Preprocessing & Normalization

Raw image inputs consist of grayscale $28 	imes 28$ pixel grids. 
* **Normalization Formula:** $X_{normalized} = rac{X}{255.0}$
* Ensures uniform feature scaling, preventing numerical instability and speeding up network convergence during backpropagation.

---

## 🤖 Neural Network Architecture

### Sequential Model Configuration
* **Input Layer:** `keras.layers.Flatten(input_shape=(28, 28))`
* **Hidden Layer:** `keras.layers.Dense(50, activation='relu')`
* **Output Layer:** `keras.layers.Dense(10, activation='softmax')`

* **Optimizer:** `adam`
* **Loss Function:** `sparse_categorical_crossentropy`
* **Evaluation Metric:** `accuracy`

---

## 📊 Visualizations Included

The script generates a dual-panel `matplotlib` chart to monitor training dynamics:
- **Accuracy Evolution Plot:** Tracks the progression of training accuracy across epochs.
- **Loss (Perda) Evolution Plot:** Tracks the minimization of training loss over successive training cycles.
