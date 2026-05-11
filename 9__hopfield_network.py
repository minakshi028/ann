
import numpy as np

# -----------------------------
# Step 1: Define Patterns (4 vectors)
# -----------------------------

patterns = np.array([
    [1, -1,  1, -1],
    [-1, 1, -1,  1],
    [1,  1, -1, -1],
    [-1, -1, 1,  1]
])

# -----------------------------
# Step 2: Training (Hebbian Rule)
# -----------------------------

n = patterns.shape[1]
W = np.zeros((n, n))

for p in patterns:
    p = p.reshape(-1, 1)
    W += np.dot(p, p.T)

# Remove self-connections
np.fill_diagonal(W, 0)

print("Weight Matrix W:\n", W)

# -----------------------------
# Step 3: Activation Function
# -----------------------------

def sign(x):
    return np.where(x >= 0, 1, -1)

# -----------------------------
# Step 4: Recall Function
# -----------------------------

def recall(input_pattern, W, steps=5):
    x = input_pattern.copy()

    for _ in range(steps):
        x = sign(np.dot(W, x))

    return x

# -----------------------------
# Step 5: Testing
# -----------------------------

print("\n--- Testing Recall ---\n")

# Noisy version of first pattern
test = np.array([1, -1, -1, -1])  # slightly corrupted

print("Input (Noisy): ", test)

output = recall(test, W)

print("Recalled Pattern:", output)

