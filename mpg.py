# Import libraries
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



# 1. Activation Function

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass
# 2. subclassing Activation Function
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


# 3.  Loss Functions


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass
# 4. subclassing loss function
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


# 5. Define Layer Class


class Layer:
    def __init__(self, fan_in, fan_out, activation_function,dropout_rate=0.0):
        # 6.  initialize the weight matrix using Glorot uniform
        self.weights = np.random.uniform(-np.sqrt(6 / (fan_in + fan_out)), np.sqrt(6 / (fan_in + fan_out)), (fan_in, fan_out))
        self.bias = np.zeros((1, fan_out))
        self.activation = activation_function
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def forward(self, x, training=True):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        self.output = self.activation.forward(self.z)
        # 7. implement dropout
        if training and self.dropout_rate > 0.0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.output.shape) / (1 - self.dropout_rate)
            self.output *= self.dropout_mask
        return self.output

# 7. backward
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

# 8. Define Multilayer Perceptron Class


class MultilayerPerceptron:
    def __init__(self, layers):
        self.layers = layers
        #9. forward and backward

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




# # fetch MPG dataset
auto_mpg = fetch_ucirepo(id=9)

# data (as pandas dataframes)
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Combine features and target into one DataFrame for easy filtering
data = pd.concat([X, y], axis=1)

# Drop rows where the target variable is NaN
cleaned_data = data.dropna()

# Split the data back into features (X) and target (y)
X = cleaned_data.iloc[:, :-1]
y = cleaned_data.iloc[:, -1]

# Do a 70/30 split (e.g., 70% train, 30% other)
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,    # for reproducibility
    shuffle=True,       # whether to shuffle the data before splitting
)

# Split the remaining 30% into validation/testing (15%/15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)


# Compute statistics for X (features)
X_mean = X_train.mean(axis=0)  # Mean of each feature
X_std = X_train.std(axis=0)    # Standard deviation of each feature


# Standardize X
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Compute statistics for y (targets)
y_mean = y_train.mean()  # Mean of target
y_std = y_train.std()    # Standard deviation of target

# Standardize y
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std
# 6. Train the Vehicle MPG Model


# Define MLP architecture for Vehicle MPG
layers = [
    Layer(fan_in=X_train.shape[1], fan_out=64, activation_function=Relu()),
    Layer(fan_in=64, fan_out=32, activation_function=Relu()),
    Layer(fan_in=32, fan_out=1, activation_function=Linear())
]

# Instantiate MLPd
mlp = MultilayerPerceptron(layers)

# Train the model
loss_func = SquaredError()
train_losses, val_losses = mlp.train(X_train.values, y_train.values.reshape(-1, 1), X_val.values, y_val.values.reshape(-1, 1), loss_func, learning_rate=0.01, epochs=100)

# Plot Training and Validation Loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Vehicle MPG Training and Validation Loss')
plt.legend()
plt.show()

# Evaluate on Test Set
y_pred_test = mlp.forward(X_test.values)
test_loss = loss_func.loss(y_test.values.reshape(-1, 1), y_pred_test)
print(f"Total Testing Loss (Vehicle MPG): {test_loss:.4f}")

# Report Predictions for 10 Samples
samples = np.random.choice(len(y_test), 10, replace=False)
predicted_mpg = y_pred_test[samples] * y_std + y_mean
true_mpg = y_test.values[samples] * y_std + y_mean

results = pd.DataFrame({"True MPG": true_mpg.flatten(), "Predicted MPG": predicted_mpg.flatten()})
print(results)
