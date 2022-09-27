import numpy as np


class ds_gmms:
    Mu = None
    Sigma = None
    Priors = None


class Vxf0_struct:
    d = 0
    w = 0
    L = 0
    Mu = None
    Priors = None
    P = None
    SOS = False


class options_struct:
    tol_mat_bias = 0
    display = 1
    tol_stopping = 0
    max_iter = 0
    optimizePriors = False
    upperBoundEigenValue = True


class Vxf_struct:
    d = 0
    w = 0
    L = 0
    Mu = None
    Priors = None
    P = None