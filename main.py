from phys_gmm_python.fit_gmm import fig_gmm
import numpy as np
from datasets.load_dataset_DS import load_dataset_DS
from utils_DS.figure_tools.plot_reference_trajectories_DS import plot_reference_trajectories_DS
from phys_gmm_python.Structs import Est_options
from utils_DS.diy_tools.knn_search import knn_search
from gmr import GMM
from phys_gmm_python.utils.plotting.simple_classification_check import plot_result
from Structs_DS import ds_gmms
from utils.adjust_Covariances import adjust_Covariances
from lpv_opt.optimize_lpv_ds_from_data import optimize_lpv_ds_from_data
from test_data_from_ml.load_learned_data_from_ML import load_learned_data_from_ML
from test_data_from_ml.port_data_to_yaml import port_data_to_yaml
import scipy as sci
from utils.plotting.plot_ellopsoid import plot_result_3D
from lpv_opt.my_learn_function_3 import my_learn_function

pkg_dir = 'E:\ds-opt-python\ds-opt-python'
chosen_dataset = 7 #6 # 4 (when conducting 2D test)
sub_sample = 2  # '>2' for real 3D Datasets, '1' for 2D toy datasets
nb_trajectories = 4  # Only for real 3D data
Data, Data_sh, att, x0_all, data, dt = load_dataset_DS(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
vel_samples = 10
vel_size = 20
# plot_reference_trajectories_DS(Data, att, vel_samples, vel_size)

# S.P.E.C.I.A.L.
P_test = my_learn_function(Data_sh)

M = int(len(Data) / 2)
Xi_ref = Data[0:M, :]
Xi_dot_ref = Data[M:, :]
# axis_limits = axis

##################################################
# Step 2 (GMM FITTING): Fit GMM to Trajectory Data#
##################################################

# 0: Physically-Consistent Non-Parametric (Collapsed Gibbs Sampler)
# 1: GMM-EM Model Selection via BIC
# 2: CRP-GMM (Collapsed Gibbs Sampler)
est_options = Est_options()
est_options.type = 0  # GMM Estimation Algorithm Type
# If algo 1 selected:
est_options.maxK = 10
est_options.fixed_K = []

# If algo 0 or 2 selected:
est_options.samplerIter = 50
est_options.do_plots = 1
# Size of sub-sampling of trajectories
# 1/2 for 2D datasets, >2/3 for real
nb_data = len(Data[0])
sub_sample = 1
if nb_data > 500:
    sub_sample = 2
elif nb_data > 1000:
    sub_sample = 3

est_options.sub_sample = sub_sample
# Metric Hyper-parameters
est_options.estimate_l = 1
est_options.l_sensitivity = 2
est_options.length_scale = []
save_result = 0
# instead of wasting your time, you could save the result directly and load them in next run for debug
if save_result:
    Priors, Mu, Sigma = fig_gmm(Xi_ref, Xi_dot_ref, est_options)
    np.save('Priors.npy', Priors)
    np.save('Mu.npy', Mu)
    np.save('Sigma.npy', Sigma)
else:
    Priors = np.load('Priors.npy')
    Mu = np.load('Mu.npy')
    Sigma = np.load('Sigma.npy')

# this just preventing
    # Generate GMM data structure for DS learning
    # This re-ordering needed to linearize the linear DS @ attracto
idx = knn_search(Mu.T, att.reshape(len(att)), len(Mu[0]))
Priors_old = Priors.copy()
Mu_old = Mu.copy()
Sigma_old = Sigma.copy()
for i in np.arange(len(idx)):
    Priors[idx[i]] = Priors_old[i]
    Mu[:, idx[i]] = Mu_old[:, i]
    Sigma[idx[i]] = Sigma_old[i]
# Make the closest Gaussian isotropic and place it at the attractor location
Sigma[0] = 1 * np.max(np.diag(Sigma[0])) * np.eye(M)
Mu[:, 0] = att.reshape(len(att))
gmm = GMM(len(Mu[0]), Priors, Mu.T, Sigma)

# (Recommended!) Step 2.1: Dilate the Covariance matrices that are too thin
# This is recommended to get smoother streamlines/global dynamics
ds_gmm = ds_gmms()
ds_gmm.Mu = Mu
ds_gmm.Sigma = Sigma
ds_gmm.Priors = Priors
adjusts_C = 1
if adjusts_C == 1:
    if M == 2:
        tot_dilation_factor = 1
        rel_dilation_fact = 0.25
    else:
        # this is for M == 3
        tot_dilation_factor = 1
        rel_dilation_fact = 0.75
    Sigma_ = adjust_Covariances(ds_gmm.Priors, ds_gmm.Sigma, tot_dilation_factor, rel_dilation_fact)
    ds_gmm.Sigma = Sigma_

# just check the result of adjust Covariances
# if M == 3:
#     plot_result_3D(ds_gmm.Mu, ds_gmm.Sigma, Xi_ref)
# elif M == 2:
#     gmm = GMM(len(ds_gmm.Priors), ds_gmm.Priors, ds_gmm.Mu.T, ds_gmm.Sigma)
#     plot_result(Xi_ref, gmm, len(ds_gmm.Mu), ds_gmm.Mu, 2)

# ds_gmm.Mu, ds_gmm.Priors, ds_gmm.Sigma, P_opt = load_learned_data_from_ML()

#############################################################
# Step 3 (DS ESTIMATION): ESTIMATE SYSTEM DYNAMICS MATRICES #
#############################################################
# DS OPTIMIZATION OPTIONS
# Type of constraints/optimization
lyap_constr = 2  # % 0:'convex':     A' + A < 0 (Proposed in paper)
                 # 2:'non-convex': A'P + PA < -Q given P (Proposed in paper)
init_cvx = 1  # 为了只跑一次可以改成0，目前这玩意好像可以不去用
symm_constr = 0

if lyap_constr == 0 or lyap_constr == 1:
    P_opt = np.eye(M)
else:
    # P_opt = P_opt
    P_opt = my_learn_function(Data_sh)

if lyap_constr == 1:
    A_k, b_k, _ = optimize_lpv_ds_from_data(Data_sh, np.eye(M), lyap_constr, ds_gmm, P_opt, init_cvx, symm_constr)
    np.save('A_k.npy', A_k)
    np.save('b_k.npy', b_k)
else:
    save_opt_result = 1
    if save_opt_result:
        A_k, b_k, _ = optimize_lpv_ds_from_data(Data, att, lyap_constr, ds_gmm, P_opt, init_cvx, symm_constr)
        np.save('A_k.npy', A_k)
        np.save('b_k.npy', b_k)
    else:
        A_k = np.load('A_k.npy')
        b_k = np.load('b_k.npy')
        b_k_test = []
        A_k_test = []
        for k in np.arange(len(ds_gmm.Sigma)):
            b_k_test.append(A_k[k] @ att * (-1))
            A_k_test.append(np.linalg.eig(P_opt @ A_k[k] + A_k[k].T @ P_opt)[0])
        A_k_test = np.array(A_k_test)
        b_k_test = np.array(b_k_test)
        calculate_velo = np.zeros((M, len(Xi_ref[0])))
        for k in np.arange(len(ds_gmm.Sigma)):
            b_k_helper = np.repeat(b_k[:, k].reshape(len(b_k[:, k]),1), len(Xi_ref[0]), axis=1)
            calculate_velo += (A_k[k] @ Xi_ref + b_k_helper) * ds_gmm.Priors[k]
        port_data_to_yaml('haruhi', ds_gmm, A_k, att, x0_all, dt)
        sci.io.savemat('A_k_b_k.mat', {'A_k': A_k, 'b_k': b_k})
        stop_bar = 1

