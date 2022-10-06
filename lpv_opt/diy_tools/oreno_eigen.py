from math import *
import numpy as np


def calculate_eigenvalue(a, b, c, d, e, f):
    x_1 = a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c + 3 * (d ** 2 + f ** 2 + e ** 2)
    x_2 = (-(2 * a - b - c) * (2 * b - a - c) * (2 * c - a - b) + 9 * (
                (d ** 2) * (2 * c - a - b) + (f ** 2) * (2 * b - a - c) + (e ** 2) * (2 * a - b - c)) +
           - 54 * d * e * f)

    if x_2 > 0:
        theta = atan(sqrt(4*x_1**3 - x_2**2) / x_2)
    elif x_2 < 0:
        theta = atan(sqrt(4 * x_1 ** 3 - x_2 ** 2) / x_2) + pi
    else:
        theta = pi/2

    lambda_1 = (a + b + c - 2 * sqrt(x_1) * cos(theta/3)) / 3
    lambda_2 = (a + b + c + 2 * sqrt(x_1) * cos((theta - pi) / 3)) / 3
    lambda_3 = (a + b + c + 2 * sqrt(x_1) * cos((theta + pi) / 3)) / 3

    return lambda_1, lambda_2, lambda_3


if __name__ == '__main__':
    print(calculate_eigenvalue(1, 2, 3, 1, 1, 1))
    print(np.linalg.eigvals([[1, 1, 1], [1, 2, 1],[1,1, 3]]))