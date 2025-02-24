# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import random
import struct
from array import array
from os.path import join
import os



# 2. Define Activation Functions


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sig = self.forward(x)
        return sig * (1 - sig)

class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x)**2

class Relu(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)

class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)

class Softplus(ActivationFunction):
    def forward(self, x):
        return np.log(1 + np.exp(x))

    def derivative(self, x):
        return 1 / (1 + np.exp(-x))

class Mish(ActivationFunction):
    def forward(self, x):
        return x * np.tanh(np.log(1 + np.exp(x)))

    def derivative(self, x):
        sp = np.log(1 + np.exp(x))
        tanh_sp = np.tanh(sp)
        return tanh_sp + x * (1 - tanh_sp ** 2) * (1 / (1 + np.exp(-x)))


# 3. Define Loss Functions


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass

class SquaredError(LossFunction):
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

    def derivative(self, y_true, y_pred):
        return -(y_true / (y_pred + 1e-9))


# 4. Define Layer Class


class Layer:
    def __init__(self, fan_in, fan_out, activation_function,dropout_rate=0.0):
        self.weights = np.random.uniform(-np.sqrt(6 / (fan_in + fan_out)), np.sqrt(6 / (fan_in + fan_out)), (fan_in, fan_out))
        self.bias = np.zeros((1, fan_out))
        self.activation = activation_function
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def forward(self, x, training=True):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        self.output = self.activation.forward(self.z)
        
        if training and self.dropout_rate > 0.0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.output.shape) / (1 - self.dropout_rate)
            self.output *= self.dropout_mask
        return self.output


    def backward(self, delta, learning_rate):
        activation_derivative = self.activation.derivative(self.z)
        delta *= activation_derivative

        grad_weights = np.dot(self.input.T, delta)
        grad_bias = np.sum(delta, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        # Propagate delta to previous layer
        return np.dot(delta, self.weights.T)

# 5. Define Multilayer Perceptron Class


class MultilayerPerceptron:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, learning_rate):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)

    def train(self, X_train, y_train, X_val, y_val, loss_func, learning_rate=0.01, epochs=100, batch_size=32, rmsprop=False):
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            # Mini-batch training
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                X_batch, y_batch = X_train[start:end], y_train[start:end]

                y_pred = self.forward(X_batch)
                loss_grad = loss_func.derivative(y_batch, y_pred)
                self.backward(loss_grad, learning_rate)

            # Compute loss for training and validation
            train_loss = loss_func.loss(y_train, self.forward(X_train))
            val_loss = loss_func.loss(y_val, self.forward(X_val))

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            
            optimizer_used = "RMSProp" if rmsprop else "SGD"
            print(f"Epoch {epoch + 1}/{epochs} ({optimizer_used}), Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        return train_losses, val_losses





# 7. Fetch and Prepare the MNIST Data


# Set file paths based on added MNIST Datasets
input_path = 'c:/Users/pnaghs1/OneDrive - Louisiana State University/github/project1'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

# MNIST Data Loader Class
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = array("B", file.read())
        
        images = np.array(image_data).reshape(size, rows * cols) / 255.0
        labels = np.array(labels)
        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

# Load MNIST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Split MNIST training data into training and validation
X_train_mnist, X_val_mnist, y_train_mnist, y_val_mnist = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# 8. Train the MNIST Model


# Convert labels to one-hot encoding
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_mnist_encoded = one_hot_encode(y_train_mnist)
y_val_mnist_encoded = one_hot_encode(y_val_mnist)
y_test_mnist_encoded = one_hot_encode(y_test)

# Define MLP architecture for MNIST
layers_mnist = [
    Layer(fan_in=784, fan_out=128, activation_function=Relu()),
    Layer(fan_in=128, fan_out=64, activation_function=Relu()),
    Layer(fan_in=64, fan_out=10, activation_function=Sigmoid())
]

# Instantiate MLP
mlp_mnist = MultilayerPerceptron(layers_mnist)

# Train the model
loss_func_mnist = CrossEntropy()
train_losses_mnist, val_losses_mnist = mlp_mnist.train(X_train_mnist, y_train_mnist_encoded, X_val_mnist, y_val_mnist_encoded, loss_func_mnist, learning_rate=0.01, epochs=50)

# Plot Training and Validation Loss
plt.plot(train_losses_mnist, label='Training Loss')
plt.plot(val_losses_mnist, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MNIST Training and Validation Loss')
plt.legend()
plt.show()

# Evaluate on Test Set
y_pred_test_mnist = mlp_mnist.forward(x_test)
accuracy = np.mean(np.argmax(y_pred_test_mnist, axis=1) == y_test)
print(f"Test Accuracy (MNIST): {accuracy * 100:.2f}%")

# Display a sample from each class
fig, axs = plt.subplots(2, 5, figsize=(10, 5))
axs = axs.flatten()
for digit in range(10):
    idx = np.where(y_test == digit)[0][0]
    axs[digit].imshow(x_test[idx].reshape(28, 28), cmap='gray')
    axs[digit].set_title(f'Predicted: {np.argmax(y_pred_test_mnist[idx])}')
    axs[digit].axis('off')
plt.tight_layout()
plt.show()
