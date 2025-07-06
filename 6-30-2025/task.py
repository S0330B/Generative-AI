import numpy as np

input = np.array([[10000], [2], [3]])


weight_hidden = np.array([[0.1, 0.2, 0.3],
                          [0.4, -0.6, 0.8]])

bias_hidden = np.array([[1], 
                        [3]])


hidden_outputs = np.dot(weight_hidden, input) + bias_hidden

def relu(x):
    return np.maximum(0, x)

hidden_activated = relu(hidden_outputs)

weight_output = np.array([[0.5, 0.9]])

bias_output = np.array([[2]])

output = np.dot(weight_output, hidden_activated) + bias_output

final_output = relu(output)

print(final_output)
