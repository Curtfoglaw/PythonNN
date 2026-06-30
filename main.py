import numpy as np

input_layer = np.random.randn(10)

hidden_layer_weights = np.random.randn(5, 10) * 0.01
hidden_bias = np.zeros(5)

output_layer_weights = np.random.randn(10, 5) * 0.01
output_bias = np.zeros(10)

def forward_pass():

    z = hidden_layer_weights.dot(input_layer)
    z = z + hidden_bias
    
    print(z)

forward_pass()