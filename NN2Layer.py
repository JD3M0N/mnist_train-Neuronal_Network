import numpy as np

# Load training and test data, skipping the header row
train_data = np.loadtxt('mnist_train.csv', delimiter=',', skiprows=1)
test_data = np.loadtxt('mnist_test.csv', delimiter=',', skiprows=1)

# Separate features and labels
X_train = train_data[:, 1:]  # Training features
y_train = train_data[:, 0]   # Training labels

X_test = test_data[:, 1:]    # Test features
y_test = test_data[:, 0]     # Test labels

# Normalize data between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Standardize data (optional but recommended)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0) + 1e-8  # Add a small constant to avoid division by zero
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std  # Use the same mean and standard deviation as in the training set

# One-hot encoding of labels
def one_hot_encode(y):
    one_hot = np.zeros((y.size, 10))
    one_hot[np.arange(y.size), y.astype(int)] = 1
    return one_hot

y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)

# Define activation functions
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Avoid overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Initialize weights and biases
def init_weights(layer_sizes):
    np.random.seed(42)  # Seed for reproducibility
    parameters = {}
    L = len(layer_sizes) - 1  # Number of layers
    for l in range(1, L + 1):
        parameters[f'W{l}'] = np.random.randn(layer_sizes[l-1], layer_sizes[l]) * np.sqrt(2 / layer_sizes[l-1])
        parameters[f'b{l}'] = np.zeros((1, layer_sizes[l]))
    return parameters

# Forward propagation
def forward_propagation(X, parameters):
    cache = {'A0': X}
    L = len(parameters) // 2  # Number of layers
    A = X
    for l in range(1, L):
        Z = np.dot(A, parameters[f'W{l}']) + parameters[f'b{l}']
        A = relu(Z)
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
    # Output layer
    ZL = np.dot(A, parameters[f'W{L}']) + parameters[f'b{L}']
    AL = softmax(ZL)
    cache[f'Z{L}'] = ZL
    cache[f'A{L}'] = AL
    return AL, cache

# Cost function with L2 regularization
def compute_cost(AL, Y, parameters, lambd):
    m = Y.shape[0]
    cross_entropy_cost = -np.sum(Y * np.log(AL + 1e-8)) / m
    L2_regularization_cost = (lambd / (2 * m)) * sum([np.sum(np.square(parameters[f'W{l}'])) for l in range(1, len(parameters) // 2 + 1)])
    cost = cross_entropy_cost + L2_regularization_cost
    return cost

# Backward propagation
def backward_propagation(Y, cache, parameters, lambd):
    gradients = {}
    m = Y.shape[0]
    L = len(parameters) // 2  # Number of layers

    # Initialize delta
    AL = cache[f'A{L}']
    dZL = AL - Y
    gradients[f'dW{L}'] = np.dot(cache[f'A{L-1}'].T, dZL) / m + (lambd / m) * parameters[f'W{L}']
    gradients[f'db{L}'] = np.sum(dZL, axis=0, keepdims=True) / m

    # Backward propagation for layers L-1 to 1
    for l in reversed(range(1, L)):
        dA = np.dot(dZL, parameters[f'W{l+1}'].T)
        dZ = dA * relu_derivative(cache[f'Z{l}'])
        gradients[f'dW{l}'] = np.dot(cache[f'A{l-1}'].T, dZ) / m + (lambd / m) * parameters[f'W{l}']
        gradients[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True) / m
        dZL = dZ

    return gradients

# Update parameters using Adam
def update_parameters(parameters, gradients, optimizer_params):
    L = len(parameters) // 2  # Number of layers
    beta1 = optimizer_params['beta1']
    beta2 = optimizer_params['beta2']
    epsilon = optimizer_params['epsilon']
    t = optimizer_params['t']
    learning_rate = optimizer_params['learning_rate']

    for l in range(1, L + 1):
        # First moment estimates
        optimizer_params[f'VdW{l}'] = beta1 * optimizer_params[f'VdW{l}'] + (1 - beta1) * gradients[f'dW{l}']
        optimizer_params[f'Vdb{l}'] = beta1 * optimizer_params[f'Vdb{l}'] + (1 - beta1) * gradients[f'db{l}']
        # Second moment estimates
        optimizer_params[f'SdW{l}'] = beta2 * optimizer_params[f'SdW{l}'] + (1 - beta2) * (gradients[f'dW{l}'] ** 2)
        optimizer_params[f'Sdb{l}'] = beta2 * optimizer_params[f'Sdb{l}'] + (1 - beta2) * (gradients[f'db{l}'] ** 2)
        # Bias correction
        VdW_corrected = optimizer_params[f'VdW{l}'] / (1 - beta1 ** t)
        Vdb_corrected = optimizer_params[f'Vdb{l}'] / (1 - beta1 ** t)
        SdW_corrected = optimizer_params[f'SdW{l}'] / (1 - beta2 ** t)
        Sdb_corrected = optimizer_params[f'Sdb{l}'] / (1 - beta2 ** t)
        # Update parameters
        parameters[f'W{l}'] -= learning_rate * VdW_corrected / (np.sqrt(SdW_corrected) + epsilon)
        parameters[f'b{l}'] -= learning_rate * Vdb_corrected / (np.sqrt(Sdb_corrected) + epsilon)

    return parameters, optimizer_params

# Initialize parameters for Adam
def initialize_optimizer_params(parameters):
    optimizer_params = {
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        't': 0,
        'learning_rate': 0.001  # Smaller learning rate for Adam
    }
    L = len(parameters) // 2  # Number of layers
    for l in range(1, L + 1):
        optimizer_params[f'VdW{l}'] = np.zeros_like(parameters[f'W{l}'])
        optimizer_params[f'Vdb{l}'] = np.zeros_like(parameters[f'b{l}'])
        optimizer_params[f'SdW{l}'] = np.zeros_like(parameters[f'W{l}'])
        optimizer_params[f'Sdb{l}'] = np.zeros_like(parameters[f'b{l}'])
    return optimizer_params

# Generate mini-batches
def get_mini_batches(X, Y, batch_size):
    m = X.shape[0]
    indices = np.random.permutation(m)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        Y_batch = Y_shuffled[i:i+batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches

# Training function
def train(X, Y, layer_sizes, epochs, lambd, batch_size):
    # Initialize parameters
    parameters = init_weights(layer_sizes)
    optimizer_params = initialize_optimizer_params(parameters)

    for epoch in range(1, epochs + 1):
        optimizer_params['t'] += 1
        mini_batches = get_mini_batches(X, Y, batch_size)
        epoch_cost = 0

        for X_batch, Y_batch in mini_batches:
            # Forward propagation
            AL, cache = forward_propagation(X_batch, parameters)
            # Compute cost
            cost = compute_cost(AL, Y_batch, parameters, lambd)
            epoch_cost += cost * X_batch.shape[0]
            # Backward propagation
            gradients = backward_propagation(Y_batch, cache, parameters, lambd)
            # Update parameters
            parameters, optimizer_params = update_parameters(parameters, gradients, optimizer_params)

        epoch_cost /= X.shape[0]
        # Print cost every certain number of epochs
        if epoch % 10 == 0:
            print(f"Cost after epoch {epoch}: {epoch_cost:.4f}")

    return parameters

# Prediction function
def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = np.argmax(AL, axis=1)
    return predictions

# Train the model
layer_sizes = [784, 128, 64, 10]  # Added an additional hidden layer
epochs = 100
lambd = 0.001  # L2 regularization factor
batch_size = 64  # Mini-batch size

parameters = train(X_train, y_train_encoded, layer_sizes, epochs, lambd, batch_size)

# Model evaluation
train_predictions = predict(X_train, parameters)
test_predictions = predict(X_test, parameters)

train_accuracy = np.mean(train_predictions == y_train) * 100
test_accuracy = np.mean(test_predictions == y_test) * 100

print(f"Training set accuracy: {train_accuracy:.2f}%")
print(f"Test set accuracy: {test_accuracy:.2f}%")
