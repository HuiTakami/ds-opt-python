import numpy as np
import cvxpy as cp


# 算导数和当前点的能量
def compute_Energy(X, Vxf):
    d = len(X)
    V, dV = GMR_Lyapunov(X, Vxf.P)
    Vdot = dV
    return V, Vdot


def GMR_Lyapunov(x, P):
    d = len(x)
    L = len(P) - 1

    P_cur = P[0]
    V_0 = np.sum(x * (P_cur @ x), axis=0, keepdims=True)
    V = V_0
    Vx = (P_cur + P_cur.T) @ x  # derivative of Lyapunov
    return V, Vx