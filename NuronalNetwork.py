import numpy as np
import pickle  # For loading the saved model
from PIL import Image  # For image processing
import os  # For file handling

# --- Load the trained model ---

# Load the model data from the pickle file
with open('trained_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Extract parameters and normalization values
parameters = model_data['parameters']
mean = model_data['mean']
std = model_data['std']

# --- Define necessary functions ---

# Activation functions
def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Prevent overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Forward propagation function
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

# Prediction function
def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = np.argmax(AL, axis=1)
    return predictions

# --- Load and preprocess images from the 'tests' folder ---

def load_and_preprocess_images(folder_path, mean, std):
    images_list = []
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for file_name in image_files:
        # Open image
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path)

        # Convert to grayscale
        image = image.convert('L')  # 'L' mode for (8-bit pixels, black and white)

        # Resize to 28x28 pixels (if not already)
        image = image.resize((28, 28))

        # Convert to numpy array
        image_array = np.array(image)

        # Invert colors if necessary (black background to white background)
        # Since MNIST digits are white on black background,
        # you might need to invert your images if they are black digits on white background.
        # Uncomment the following line if needed:
        # image_array = 255 - image_array

        # Flatten the image (MNIST images are flattened to 784 features)
        image_flatten = image_array.flatten()

        # Normalize the image between 0 and 1
        image_normalized = image_flatten / 255.0

        # Standardize the image using the training set's mean and std
        image_standardized = (image_normalized - mean) / std

        images_list.append(image_standardized)
    X_new = np.array(images_list)
    return X_new, image_files  # Also return the file names

# Specify the folder containing the test images
folder_path = 'tests'  # Replace with your folder path if different

# Load and preprocess images
X_new, image_files = load_and_preprocess_images(folder_path, mean, std)

# --- Make predictions on the new images ---

# Ensure that there are images to predict
if X_new.size == 0:
    print("No images found in the specified folder.")
else:
    # Predict using the loaded model
    new_predictions = predict(X_new, parameters)

    # Print the predictions
    for file_name, pred in zip(image_files, new_predictions):
        print(f"Image {file_name} is predicted as {pred}")
