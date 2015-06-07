import numpy as np

# Sigmoid and cost function derivatives
def sigmoid_transfer(x):
    return 1.0/(1.0+np.exp(-x))
sigmoid_transfer = np.vectorize(sigmoid_transfer)

def sigmoid_transfer_deriv(x):
    return sigmoid_transfer(x)*(1-sigmoid_transfer(x))
sigmoid_transfer_deriv = np.vectorize(sigmoid_transfer_deriv)

def linear_transfer(x):
    return x;
linear_transfer = np.vectorize(linear_transfer)

def linear_transfer_deriv(x):
    return 1;
linear_transfer_deriv = np.vectorize(linear_transfer_deriv)

def cost_MSE_deriv(outputs, cost):
    return outputs-cost
cost_MSE_deriv = np.vectorize(cost_MSE_deriv)

### ASSERT TESTING ###

# Sigmoid transfer
assert(np.allclose(sigmoid_transfer([-0.5, 0, 1]), [0.3775, 0.5, 0.7311], 0.01))
assert(sigmoid_transfer(-100) >= 0 and sigmoid_transfer(100) <= 1)
assert(np.allclose(sigmoid_transfer_deriv([-0.5, 0, 1]), [0.235, 0.25, 0.1966], 0.001))

# Linear transfer
assert(np.allclose(linear_transfer([1, 100, -1000]), [1, 100, -1000], 0))
assert(np.allclose(linear_transfer_deriv([1, 100, -1000]), [1, 1, 1], 0))

# MSE cost function
assert(np.allclose(cost_MSE_deriv([1, 0], [0, -1]), [1, 1], 0))