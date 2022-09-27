import numpy as np
from Structs_DS import Vxf0_struct, options_struct
from learnEnergy import learn_energy

def learn_wsaqf(Data, *args):
    Vxf0 = Vxf0_struct()
    Vxf0.d = int(len(Data) / 2)
    Vxf0.w = 10 ** (
        -4)  # A positive scalar weight regulating the priority between the two objectives of the opimization
    options = options_struct()
    # options passed to solver
    options.tol_mat_bias = 10 ** (-1)
    options.display = 1
    options.tol_stopping = 10 ** -10
    options.max_iter = 1000
    options.optimizePriors = False
    options.upperBoundEigenValue = True

    # Initial Guess for WSAQF Parameters

    if len(args) >= 2:
        print('we currently dont offer this function, Σ(っ °Д °;)っ')
        return None

    else:
        Vxf0.L = 0
        Vxf0.Mu = np.zeros((Vxf0.d, Vxf0.L + 1))
        Vxf0.Priors = np.ones((Vxf0.L + 1, 1))
        Vxf0.Priors = Vxf0.Priors / sum(Vxf0.Priors)
        Vxf0.P = np.zeros((Vxf0.L + 1, Vxf0.d, Vxf0.d))
        for l in np.arange(Vxf0.L + 1):
            Vxf0.P[l] = np.eye(Vxf0.d)

        # initialization complete, ready to optimize
        Vxf = learn_energy(Vxf0, Data, options)
        return Vxf