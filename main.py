import numpy as np

input_layer = np.random.randn(10)

hidden_layer_weights = np.random.randn(5, 10) * 0.01
hidden_bias = np.zeros(5)

output_layer_weights = np.random.randn(10, 5) * 0.01
output_bias = np.zeros(10)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - max(x))
    return exp_x / np.sum(exp_x)

def forward_pass():

    z_h = hidden_layer_weights.dot(input_layer)
    z_h = z_h + hidden_bias
    z_h = relu(z_h)
    
    z_o = output_layer_weights.dot(z_h)
    z_o = z_o + output_bias
    print(z_o)

forward_pass()