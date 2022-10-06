import numpy as np
import cvxpy as cp


# reconstruction, ready to
def my_learn_function(Data):
    d = int(len(Data) / 2)
    x = Data[:d, :]
    xd = Data[d:, :]
    P = cp.Variable((d, d))
    constrains = [P >> 0]
    prob = cp.Problem(cp.Minimize(object_function(P, x, xd, d, 0.0001)), constrains)
    prob.solve(solver=cp.MOSEK, verbose=True)
    P_0 = P.value
    return P_0


def object_function(P, x, xd, d, w):
    Vx, V_dot = compute_Energy(x, xd, P)
    norm_Vx = cp.sqrt(cp.sum(cp.multiply(Vx, Vx), axis=0))
    norm_xd = cp.sqrt(cp.sum(cp.multiply(xd, xd), axis=0))
    J = V_dot / cp.multiply(norm_Vx, norm_xd)
    J_total = (1 + w) / 2 * J[0]**3/cp.abs(J[0]) + (1 - w) / 2 * J[0]**2

    for i in np.arange(1, len(x[0])):
        J_total += (1 + w) / 2 * J[i]**3/cp.abs(J[i]) + (1 - w) / 2 * J[i]**2

    return J_total


def compute_Energy(X, Xd, P):
    dV = (P + P.T) @ X  # 3x3 @ 3xn = 3xn (derivative of lyap)
    V_dot = cp.sum(cp.multiply(Xd, dV), axis=0)  # derivative respect to time nx1
    return dV, V_dot
