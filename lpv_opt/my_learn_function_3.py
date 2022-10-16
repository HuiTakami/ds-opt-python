import scipy as sp
import numpy as np
from lpv_opt.my_check_function import my_check_function
from math import *
from casadi import *


def my_learn_function(Data):
    d = int(len(Data) / 2)
    x = Data[:d, :]
    xd = Data[d:, :]

    # opti = casadi.Opti()
    # p = opti.variable(6)
    p0 = cov_initial_guess(x)
    # opti.set_initial(p, p0)
    print('the target function originally is ', my_check_function(Data, p0))
    print('the eigenvalue of initial guess is ', np.linalg.eigvals(vector_to_matrix(p0)))
    p = SX.sym('y', 6)
    lambda_1, lambda_2, lambda_3 = calculate_eigenvalue(p)
    g = vertcat(lambda_1 - 0.1, lambda_2 - 0.1, lambda_3 - 0.1)
    nlp = {'x': p, 'f': object_function(p, x, xd, 0.0001), 'g': g}
    S = nlpsol('S', 'ipopt', nlp)
    r = S(x0=p0,
          lbg=[0.00, 0.00, 0.00], ubg=[inf, inf, inf])
    x_opt = r['x']
    print('x_opt: ', x_opt)
    # lambda_1, lambda_2, lambda_3 = calculate_eigenvalue(p)
    # opti.subject_to([lambda_1 >= 0])
    # opti.subject_to([lambda_2 >= 0])
    # opti.subject_to([lambda_3 >= 0])
    # opti.subject_to([lambda_1 + lambda_2 + lambda_3 == 1])
    # p_opts = {"expand": True}
    # s_opts = {"max_iter": 1000}
    # opti.solver("ipopt", p_opts,
    #             s_opts)
    # opti.minimize(object_function(p, x, xd, 0.0001))
    # sol = opti.solve()
    # result = sol.value(x)
    # print(result)

    result = vector_to_matrix(np.array(x_opt).reshape(6))
    print("the result is", result)
    sp.io.savemat('P_python.mat', {'P': result})
    print("the eigen_value of P is", np.linalg.eigvals(result))
    if len(x) == 3:
        print('the target value finally will be ', my_check_function(Data, np.array(x_opt).reshape(6)))
    stop = 1
    return result


def object_function(P, x, xd, w):
    J_total = 0
    for i in np.arange(len(x[0])):
        dlyap_dx, dlyap_dt = compute_Energy_Single(x[:, i], xd[:, i], P)
        norm_vx = sqrt(dlyap_dx[0]**2 + dlyap_dx[1]**2 + dlyap_dx[2] ** 2)
        norm_xd = sqrt(xd[:, i][0]**2 + xd[:, i][1]**2 + xd[:, i][2]**2)

        J = if_else(logic_or(norm_xd == 0, norm_vx == 0), 0, dlyap_dt / (norm_vx * norm_xd))
        J_total += if_else(dlyap_dt < 0, -w * J**2, J ** 2)
        # if norm_xd == 0 or norm_vx == 0:
        #     J = 0
        # else:
        #     J = dlyap_dt / (norm_vx * norm_xd)
        #     if dlyap_dt < 0:
        #         J_total += -w * J**2
        #     else:
        #         J_total += J ** 2

    return J_total


def compute_Energy_Single(x, xd, p):
    # lyap resp to x (P + P.T) @ X : shape: 3
    if len(x) == 3:
        dlyap_dx_1 = 2 * (p[0] * x[0] + p[1] * x[1] + p[2] * x[2])
        dlyap_dx_2 = 2 * (p[1] * x[0] + p[3] * x[1] + p[4] * x[2])
        dlyap_dx_3 = 2 * (p[2] * x[0] + p[4] * x[1] + p[5] * x[2])
        # lyap resp to t
        v_dot = xd[0] * dlyap_dx_1 + xd[1] * dlyap_dx_2 + xd[2] * dlyap_dx_3
        # derivative of x
        dv = [dlyap_dx_1, dlyap_dx_2, dlyap_dx_3]
    else:
        dlyap_dx_1 = 2 * (p[0] * x[0] + p[1] * x[1])
        dlyap_dx_2 = 2 * (p[1] * x[0] + p[2] * x[1])
        v_dot = xd[0] * dlyap_dx_1 + xd[1] * dlyap_dx_2
        dv = [dlyap_dx_1, dlyap_dx_2]

    return dv, v_dot


def constrians(p):
    P = np.array([[p[0], p[1], p[2]], [p[1], p[3], p[4]],[p[2], p[4], p[5]]])
    return sp.linalg.eigvals(P)


def vector_to_matrix(p):
    if len(p) == 6:
        P = np.array([[p[0], p[1], p[2]], [p[1], p[3], p[4]], [p[2], p[4], p[5]]])
    else:
        P = np.array([[p[0], p[1]],
                      [p[1], p[2]]])
    return P


def calculate_eigenvalue(P):
    a, b, c, d, e, f = P[0], P[3], P[5], P[1], P[4], P[2]
    x_1 = a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c + 3 * (d ** 2 + f ** 2 + e ** 2)
    x_2 = (-(2 * a - b - c) * (2 * b - a - c) * (2 * c - a - b) + 9 * (
                (d ** 2) * (2 * c - a - b) + (f ** 2) * (2 * b - a - c) + (e ** 2) * (2 * a - b - c)) +
           - 54 * d * e * f)

    theta = if_else(x_2 > 0, atan(sqrt(4*x_1**3 - x_2**2) / x_2), if_else(x_2 < 0, atan(sqrt(4 * x_1 ** 3 - x_2 ** 2) / x_2) + pi, pi/2))

    lambda_1 = (a + b + c - 2 * sqrt(x_1) * cos(theta/3)) / 3
    lambda_2 = (a + b + c + 2 * sqrt(x_1) * cos((theta - pi) / 3)) / 3
    lambda_3 = (a + b + c + 2 * sqrt(x_1) * cos((theta + pi) / 3)) / 3

    return lambda_1, lambda_2, lambda_3


def calculate_eigenvalue_2D(P):
    a = P[0]
    b = P[2]
    c = P[1]
    gamma_ = sqrt(4*(c**2) + (a - b)**2)
    lambda_1 = (a + b - gamma_) / 2
    lambda_2 = (a + b + gamma_) / 2
    return lambda_1, lambda_2


def cov_initial_guess(data):
    cov = np.cov(data)
    print('previous eigenvalues is ', np.linalg.eigvals(cov))
    U, S, VT = np.linalg.svd(cov)
    S = S * 10
    cov = U @ np.diag(S) @ VT
    print("After expansion is ", np.linalg.eigvals(cov))

    p = [cov[0][0], cov[0][1], cov[0][2], cov[1][1], cov[1][2], cov[2][2]]
    return p


if __name__ == '__main__':
    print(calculate_eigenvalue_2D([1, 0, 2]))
