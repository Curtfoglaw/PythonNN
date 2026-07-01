import numpy as np

y_true = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
input_layer = np.random.randn(1, 10)

hidden_layer_weights = np.random.randn(10, 5) * 0.01
hidden_bias = np.zeros((1, 5))

output_layer_weights = np.random.randn(5, 10) * 0.01
output_bias = np.zeros((1, 10))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def cross_entropy(y_true, y_pred):
    loss = -np.sum(y_true * np.log(y_pred))
    return loss

def forward_pass(X):

    z_h = X.dot(hidden_layer_weights) + hidden_bias
    a_h = relu(z_h)
    
    z_o = z_h.dot(output_layer_weights) + output_bias
    a_o = softmax(z_o)
    
    return z_h, a_h, z_o, a_o 

def back_prop():

z_h, a_h, z_o, a_o = forward_pass(input_layer)

loss = cross_entropy(y_true, a_o)

print(a_o)
print(loss)
