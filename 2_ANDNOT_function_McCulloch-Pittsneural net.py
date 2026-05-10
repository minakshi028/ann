
def mcculloch_pitts_and_not(A, B):
    """
    Implements AND-NOT function using McCulloch-Pitts neuron
    Output = A AND (NOT B)
    """

    # Weights
    w1 = 1   # weight for input A
    w2 = -1  # weight for input B (negative for NOT)

    # Threshold
    threshold = 1

    # Net input
    net_input = (A * w1) + (B * w2)

    # Activation function (step function)
    if net_input >= threshold:
        return 1
    else:
        return 0


# Main program
print("McCulloch-Pitts AND-NOT Function")
print("---------------------------------")
print("A  B  |  Output")
print("----------------")

# All input combinations
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Compute and display results
for A, B in inputs:
    output = mcculloch_pitts_and_not(A, B)
    print(f"{A}  {B}  |    {output}")
