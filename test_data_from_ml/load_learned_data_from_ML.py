from scipy.io import loadmat
import numpy as np


def load_learned_data_from_ML():
    pkg_dir_mu = r'E:\ds-opt-python\ds-opt-python\test_data_from_ml\3D_Data' + '\\' + r'matlab_mu.mat'
    pkg_dir_priors = r'E:\ds-opt-python\ds-opt-python\test_data_from_ml\3D_Data' + '\\' + r'matlab_priors.mat'
    pkg_dir_sigma = r'E:\ds-opt-python\ds-opt-python\test_data_from_ml\3D_Data' + '\\' + r'matlab_sigma.mat'
    pkg_dir_P = r'E:\ds-opt-python\ds-opt-python\test_data_from_ml\3D_Data' + '\\' + r'P_opt.mat'
    data_ = loadmat(r"{}".format(pkg_dir_mu))
    Mu = np.array(data_["Mu"])
    data_ = loadmat(r"{}".format(pkg_dir_priors))
    Priors = np.array(data_["Priors"])[0]
    K = len(Priors)
    dim = len(Mu)
    data_ = loadmat(r"{}".format(pkg_dir_sigma))
    Sigma_data = np.array(data_["Sigma_"])
    Sigma_data_parts = []
    for d in np.arange(dim):
        Sigma_data_parts.append(Sigma_data[d])
    Sigma = np.zeros((K, dim, dim))
    for k in np.arange(K):
        for d in np.arange(dim):
            Sigma[k][:, d] = Sigma_data_parts[d][:, k]

    data_ = loadmat(r"{}".format(pkg_dir_P))
    P = np.array(data_['P_opt'])
    return Mu, Priors, Sigma, P

