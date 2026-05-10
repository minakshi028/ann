import numpy as np

# Convert ASCII value to 8-bit binary
def ascii_to_binary(n):
    return [int(bit) for bit in format(n, '08b')]

# Training data (ASCII of digits 0–9)
X = []
for i in range(10):
    ascii_val = ord(str(i))
    X.append(ascii_to_binary(ascii_val))

X = np.array(X)

# Target output
# Even = 0, Odd = 1
y = np.array([i % 2 for i in range(10)])

# Initialize weights and bias
weights = np.random.rand(8)
bias = np.random.rand()

# Learning rate
lr = 0.1

# Step activation function
def step(x):
    return 1 if x >= 0 else 0

# Training
epochs = 20

for epoch in range(epochs):
    for i in range(len(X)):
        net = np.dot(X[i], weights) + bias
        output = step(net)

        error = y[i] - output

        # Update weights
        weights = weights + lr * error * X[i]
        bias = bias + lr * error

# Testing
print("Digit | ASCII | Prediction")
print("---------------------------")

for i in range(10):
    x = ascii_to_binary(ord(str(i)))
    net = np.dot(x, weights) + bias
    pred = step(net)

    print(f"{i}   |  {ord(str(i))}   |   {pred}")

for i in range(10):
    x = ord(str(i)) / 100.0
    net = x * weight + bias
    pred = step(net)
    print(f"  {i}   |  {ord(str(i))}   |      {pred}")
