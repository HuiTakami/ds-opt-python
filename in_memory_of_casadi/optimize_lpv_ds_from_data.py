from lpv_opt.posterior_probs_gmm import posterior_probs_gmm
from casadi import *
import numpy as np


def optimize_lpv_ds_from_data(Data, attractor, ctr_type, gmm, *args):
    M = len(Data)
    N = len(Data[0])
    M = int(M / 2)

    # Positions and Velocity Trajectories
    Xi_ref = Data[0:M, :]
    Xi_ref_dot = Data[M:, :]

    # Define Optimization Variables
    K = len(gmm.Priors)
    A_c = np.zeros((K, M, M))
    b_c = np.zeros((M, K))

    # should have switch ctr_type here
    if ctr_type == 0:
        helper = 1  # blank for later use
        symm_constr = 0
    else:
        print('we dont currently offer this function')

    if len(args) >= 1:
        P = args[0]
        if len(args) >= 2:
            init_cvx = args[1]
            if len(args) >= 3:
                symm_constr = args[2]

    if init_cvx:
        print('Solving Optimization Problem with Convex Constraints for Non-Convex Initialization...')
        A0, b0, _ = optimize_lpv_ds_from_data(Data, attractor, 0, gmm, np.eye(M), 0, symm_constr)

    h_k = posterior_probs_gmm(Xi_ref, gmm, 'norm')

    # Define Constraints and Assign Initial Values
    # 创建一个object 叫decision variable，which makes it
    opti = casadi.Opti()
    A_vars = []
    b_vars = []
    Q_vars = []
    for k in np.arange(K):
        if symm_constr:
            A_vars.append(opti.variable(M, M, 'symmetric'))
        else:
            A_vars.append(opti.variable(M, M))

        if k == 0:
            A_vars[k] = opti.variable(M, M, 'symmetric')

        b_vars.append(opti.variable(M, 1))
        Q_vars.append(opti.variable(M, M, 'symmetric'))

        if init_cvx:
            opti.set_initial(A_vars[k], A0[k])
            opti.set_initial(b_vars[k], b0[:, k])
        else:
            opti.set_initial(A_vars[k], -np.eye(M))
            opti.set_initial(b_vars[k], -np.eye(M) @ attractor)

        epi = 0.0001
        zero_helper = opti.parameter(M, M)
        epi = epi * -np.eye(M)
        opti.set_value(zero_helper, epi)
        att = opti.parameter(M, 1)
        opti.set_value(att, attractor)
        # Define Constraints
        if ctr_type == 0:
            opti.subject_to(eig_symbolic(A_vars[k].T + A_vars[k]) <= zero_helper)
            opti.subject_to(b_vars[k] == -A_vars[k] @ att)

    # Calculate our estimated velocities caused by each local behavior
    Xi_d_dot_c_raw = []
    for i in np.arange(K):
        Xi_d_dot_c_raw.append(opti.parameter(M, N))

    for k in np.arange(K):
        h_K = np.repeat(h_k[k, :].reshape(1, len(h_k[0])), M, axis=0)
        if ctr_type == 1:
            print('this method is working on progress')
        else:
            f_k = A_vars[k] @ Xi_ref + repmat(b_vars[k], 1, N)
        h_K = MX(h_K)
        Xi_d_dot_c_raw[k] = h_K * f_k

    # Sum each of the local behaviors to generate the overall behavior at
    # each point
    Xi_dot_error = np.zeros((M,N))
    for k in np.arange(K):
        Xi_dot_error = Xi_dot_error + (Xi_d_dot_c_raw[k] - Xi_ref_dot)

    # Defining Objective Function depending on constraints
    if ctr_type == 0:
        Xi_dot_total_error = opti.parameter(1)
        opti.set_value(Xi_dot_total_error, 0)
        for n in np.arange(N):
            Xi_dot_total_error = Xi_dot_total_error + norm_2(Xi_dot_error[:, n])

    opti.minimize(Xi_dot_total_error)
    opti.solver('ipopt')
    sol = opti.solve()

    return A_c, b_c, P



