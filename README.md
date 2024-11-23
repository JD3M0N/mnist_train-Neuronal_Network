# MNIST Neural Network

This project implements a simple neural network to classify handwritten digits from the MNIST dataset. There are three implementations of the neural network:

1. **Single Layer Neural Network**: Implemented in [NN1Layer.py](NN1Layer.py).
2. **Two Layer Neural Network**: Implemented in [NN2Layer.py](NN2Layer.py).
3. **Three Layer Neural Network**: Implemented in [NN3Layer.py](NN3Layer.py).

The trained model provided in the `trained_model.pkl` file is based on the two-layer neural network implementation.

## Files

- [NN1Layer.py](NN1Layer.py): Script containing the single layer neural network implementation.
- [NN2Layer.py](NN2Layer.py): Script containing the two layer neural network implementation.
- [NN3Layer.py](NN3Layer.py): Script containing the three layer neural network implementation.
- [NuronalNetwork.py](NuronalNetwork.py): Main script for loading the trained model and making predictions.
- [mnist_train.csv](mnist_train.csv): Training dataset.
- [mnist_test.csv](mnist_test.csv): Test dataset.
- [trained_model.pkl](trained_model.pkl): Pre-trained model using the two-layer neural network.
- [tests/](tests/): Folder containing test images for prediction.
- [archive.zip](archive.zip): Contains `mnist_train.csv` and `mnist_test.csv`.

## Usage

1. Ensure that the [mnist_train.csv](mnist_train.csv) and [mnist_test.csv](mnist_test.csv) files are in the same directory as the scripts.
2. To train and evaluate the neural network, run the respective script:
    - For single layer neural network: `python NN1Layer.py`
    - For two layer neural network: `python NN2Layer.py`
    - For three layer neural network: `python NN3Layer.py`
3. To make predictions using the pre-trained model, run the [NuronalNetwork.py](NuronalNetwork.py) script:
    ```sh
    python NuronalNetwork.py
    ```

## Requirements

- numpy
- pillow

Install the required packages using:
```sh
pip install numpy pillow
```

## Detailed Explanation

### Single Layer Neural Network

The single layer neural network is implemented in [NN1Layer.py](NN1Layer.py). It consists of one hidden layer and an output layer. The script includes functions for forward propagation, backward propagation, and parameter updates.

### Two Layer Neural Network

The two layer neural network is implemented in [NN2Layer.py](NN2Layer.py). It consists of two hidden layers and an output layer. The script includes functions for forward propagation, backward propagation, parameter updates, and mini-batch gradient descent. The pre-trained model provided in `trained_model.pkl` is based on this implementation.

### Three Layer Neural Network

The three layer neural network is implemented in [NN3Layer.py](NN3Layer.py). It consists of three hidden layers and an output layer. The script includes functions for forward propagation, backward propagation, parameter updates, and mini-batch gradient descent.

### Pre-trained Model

The pre-trained model provided in `trained_model.pkl` was trained using the two layer neural network implementation. The model includes the learned parameters and normalization values (mean and standard deviation) used during training. To use the pre-trained model for making predictions, run the [NuronalNetwork.py](NuronalNetwork.py) script.

### Loading and Preprocessing Images

The [NuronalNetwork.py](NuronalNetwork.py) script includes functions for loading and preprocessing images from the `tests/` folder. The images are resized to 28x28 pixels, converted to grayscale, normalized, and standardized using the mean and standard deviation from the training set.

### Making Predictions

To make predictions on new images using the pre-trained model, place the images in the `tests/` folder and run the [NuronalNetwork.py](NuronalNetwork.py) script. The script will load the images, preprocess them, and make predictions using the pre-trained model.