import numpy as np


# Initialize
weights = np.around(np.random.uniform(size=6), decimals=2)
biases = np.around(np.random.uniform(size=3), decimals=2)

# Print
print(" ---- Welcome to Rahul's nifty neuron.... -----")
print("Initial weights:", weights)
print("Initial biases:", biases)

# Initial inputs
x_1 = 0.5
x_2 = 0.85

print(f"x1 is {x_1} and x2 is {x_2}")

# Compute z_11
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print(f"Weighted sum of initial inputs' linear combo:", z_11)

# Compute z_12
z_12 =  x_1 * weights[2] + x_2 * weights[3] + biases[1]
print(f"Weighted sum of initial inputs' linear combo:", z_12)

# Activated vals
a_11 = 1.0/(1.0 + np.exp(-z_11))
a_12 = 1.0/(1.0 + np.exp(-z_12))
print(f"Activation of 1,1 : {a_11} | Activation of 1,2 : {a_12}")
# Extremely annoyed at the example not using f-strings!!!!

# Next layer
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
a_2 = 1.0 / (1.0 + np.exp(-z_2))

##################################

# Using a more programmatic approach in nb...

