import scipy as sp
import numpy as np
from lpv_opt.my_check_function import my_check_function


def my_learn_function(Data):
    d = int(len(Data) / 2)
    x = Data[:d, :]
    xd = Data[d:, :]
    nlc = sp.optimize.NonlinearConstraint(constrians, [0, 0, 0], [np.inf, np.inf, np.inf])
    p0 = np.array([1, 0, 0, 1, 0, 1])
    res = sp.optimize.minimize(object_function, p0, args=(x, xd, 0.0001), method='trust-constr', constraints=nlc)
    result = res['x']
    result = vector_to_matrix(result)
    print(res)
    print("the result is", result)
    print("the eigen_value of P is", np.linalg.eigvals(result))
    print(my_check_function(Data, res['x']))
    return result


def object_function(P, x, xd, w):
    J_total = 0
    for i in np.arange(len(x)):
        dlyap_dx, dlyap_dt = compute_Energy_Single(x[:, i], xd[:, i], P)
        norm_vx = sp.linalg.norm(dlyap_dx, 2)
        norm_xd = sp.linalg.norm(xd[:, i], 2)
        if norm_xd == 0 or norm_vx == 0:
            J = 0
        else:
            J = dlyap_dt / ((norm_vx * norm_xd) + 0.000001)
        J_total += (1 + w) / 2 * J**2 * np.sign(J) + (1 - w) / 2 * J**2

    return J_total


def compute_Energy_Single(x, xd, p):
    # lyap resp to x (P + P.T) @ X : shape: 3
    dlyap_dx_1 = 2 * (p[0] * x[0] + p[1] * x[1] + p[2] * x[2])
    dlyap_dx_2 = 2 * (p[1] * x[0] + p[3] * x[1] + p[4] * x[2])
    dlyap_dx_3 = 2 * (p[2] * x[0] + p[4] * x[1] + p[5] * x[2])
    # lyap resp to y
    v_dot = xd[0] * dlyap_dx_1 + xd[1] * dlyap_dx_2 + xd[2] * dlyap_dx_3
    # derivative of x
    dv = np.array([dlyap_dx_1, dlyap_dx_2, dlyap_dx_3])
    return dv, v_dot


def constrians(p):
    P = np.array([[p[0], p[1], p[2]], [p[1], p[3], p[4]],[p[2], p[4], p[5]]])
    return sp.linalg.eigvals(P)


def vector_to_matrix(p):
    P = np.array([[p[0], p[1], p[2]], [p[1], p[3], p[4]], [p[2], p[4], p[5]]])
    return P