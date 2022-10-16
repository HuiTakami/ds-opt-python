import numpy as np
from lpv_opt.my_learn_function_2 import my_learn_function
import scipy as sp


# this function is used to generate fake stable dynamic system for testing optimization correctness
def generate_dynamics(initial_point, P, dt, size):
    dim = len(initial_point)
    data = np.zeros((dim*2, size))
    x = initial_point
    x_dot = (P + P.T) @ initial_point
    for i in np.arange(len(data[0])):
        x = x - x_dot * dt
        x_dot = (P + P.T) @ x
        data[:dim, i] = x
        data[dim:, i] = x_dot
    data[:, -1] = 0
    sp.io.savemat('fake_data.mat', {'data_fake': data})
    return data


if __name__ == '__main__':
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dynamic_data = generate_dynamics([30, 30, 30], P, 0.075, 100)
    my_learn_function(dynamic_data)
