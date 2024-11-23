import numpy as np

# Load the training and test data, skipping the header row
train_data = np.loadtxt('mnist_train.csv', delimiter=',', skiprows=1)
test_data = np.loadtxt('mnist_test.csv', delimiter=',', skiprows=1)

# Separating the data into features and labels
X_train = train_data[:, 1:]  # Training features
y_train = train_data[:, 0]   # Training labels

X_test = test_data[:, 1:]    # Features of test
y_test = test_data[:, 0]     # Labels of test

# Normalizing the data letting it be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# One hot encoding the labels
def one_hot_encode(y):
    one_hot = np.zeros((y.size, 10))
    one_hot[np.arange(y.size), y.astype(int)] = 1
    return one_hot

y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)

# To start, we will implement a simple neural network with one hidden layer.

# Input: 784 neurons (28x28 pixels)
# Hidden Layer: You can choose, for example, 64 neurons
# Output: 10 neurons (one for each digit)

# Initialize weights and biases with small random values.

def init_weights(input_size, hidden_size, output_size):
    np.random.seed(42)  # Seed the random number generator
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Define Activation Functions
# Sigmoide

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# ReLU (Rectified Linear Unit)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Softmax (to be used in the output layer)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract the maximum value to avoid overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Forward Propagation

def forward_propagation(X, W1, b1, W2, b2):
    # Hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    # Output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

# Cost Function: We will use categorical cross-entropy.

def compute_cost(A2, Y):
    m = Y.shape[0]
    cost = -np.sum(Y * np.log(A2 + 1e-8)) / m # Add a small value to avoid log(0)
    return cost

# Backward Propagation

def backward_propagation(X, Y, cache, W2):
    Z1, A1, Z2, A2 = cache
    m = X.shape[0]
    
    # Gradient in the output layer
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    # Gradient in the hidden layer
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

# Updating Weights using gradient descent.

def update_parameters(W1, b1, W2, b2, gradients, learning_rate):
    W1 -= learning_rate * gradients["dW1"]
    b1 -= learning_rate * gradients["db1"]
    W2 -= learning_rate * gradients["dW2"]
    b2 -= learning_rate * gradients["db2"]
    return W1, b1, W2, b2


# Training the Neural Network
# -Set Training Parameters
# Number of Epochs: For example, 1000
# Batch Size: You can start with the entire set (batch gradient descent) or implement mini-batch
# Learning Rate: For example, 0.1

# Implement the Training Loop

def train(X, Y, input_size, hidden_size, output_size, epochs, learning_rate):
    # Initialize parameters
    W1, b1, W2, b2 = init_weights(input_size, hidden_size, output_size)
    
    for i in range(epochs):
        # Forward propagation
        A2, cache = forward_propagation(X, W1, b1, W2, b2)
        
        # Compute cost
        cost = compute_cost(A2, Y)
        
        # Backward propagation
        gradients = backward_propagation(X, Y, cache, W2)
        
        # Update parameters
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, gradients, learning_rate)
        
        # Print cost every certain number of epochs
        if i % 100 == 0:
            print(f"Cost after epoch {i}: {cost}")
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

# Training the model

# Parameters
input_size = 784
hidden_size = 64
output_size = 10
epochs = 1000
learning_rate = 0.1

parameters = train(X_train, y_train_encoded, input_size, hidden_size, output_size, epochs, learning_rate)

# Model Evaluation and Making Predictions

def predict(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    A2, _ = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=1)
    return predictions

# Compute the accuracy of the model

train_predictions = predict(X_train, parameters)
test_predictions = predict(X_test, parameters)

train_accuracy = np.mean(train_predictions == y_train) * 100
test_accuracy = np.mean(test_predictions == y_test) * 100

print(f"Accuracy on the training set: {train_accuracy:.2f}%")
print(f"Accuracy on the test set: {test_accuracy:.2f}%")

