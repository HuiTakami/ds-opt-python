import cvxpy as cp
import numpy as np
from Structs_DS import Vxf_struct
from computeEnergy import compute_Energy

def learn_energy(Vxf0, Data, options):
    ### initializing ###
    # parsing options
    if options is None:
        print('finish the options setting func plz')
        return None

    d = int(len(Data) / 2)
    x = Data[:d, :]
    xd = Data[d:, :]
    Vxf0.SOS = False

    ### Optimization ###
    # transforming the Lyapunov model into a vector of optimization parameters
    if Vxf0.SOS:
        print('we currently dont offer this func')
        return None
    else:
        for l in np.arange(Vxf0.L + 1):
            print('since we got only one P here so we dont do arrangement')
            p0 = GMM_2_Parameters(Vxf0, options)


def GMM_2_Parameters(Vxf, options):
    d = Vxf.d
    p0 = Vxf.P[0].reshape((d ** 2, 1))
    return p0


def object_function(p, x, xd, d, L, w, options):
    Vxf = shape_DS(p, d, L, options)
    _, Vx = compute_Energy(x, Vxf)
    Vdot = cp.sum(cp.multiply(Vx, xd), axis=0)
    norm_Vx = cp.sqrt(cp.sum(cp.multiply(Vx, Vx), axis=0))
    norm_xd = cp.sqrt(cp.sum(cp.multiply(xd, xd), axis=0))
    J = Vdot / (norm_Vx )

def shape_DS(p, d, L, options):
    P = np.zeros((d, d))
    Priors = 1
    Mu = np.zeros((d, 1))
    i_c = 1

    P = p.reshape((d, d))
    Vxf = Vxf_struct()
    Vxf.Priors = Priors
    Vxf.Mu = Mu
    Vxf.P = P
    return Vxf

