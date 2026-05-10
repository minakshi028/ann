import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset (XOR)
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# Target output
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Initialize weights and bias
np.random.seed(42)

# Input layer to hidden layer
W1 = np.random.rand(2,2)
b1 = np.random.rand(1,2)

# Hidden layer to output layer
W2 = np.random.rand(2,1)
b2 = np.random.rand(1,1)

# Learning rate
lr = 0.5

# Number of epochs
epochs = 10000

# Training using backpropagation
for epoch in range(epochs):

    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # Error calculation
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = np.dot(d_output, W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and bias
    W2 += np.dot(hidden_output.T, d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr

    W1 += np.dot(X.T, d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

# Testing
print("XOR Results:\n")

for i in range(len(X)):
    hidden = sigmoid(np.dot(X[i], W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)

    print(f"Input: {X[i]} -> Output: {output[0][0]:.4f}")
