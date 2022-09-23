import yaml
import numpy as np
from test_data_from_ml.load_learned_data_from_ML import load_learned_data_from_ML


# this function help you to port data to yaml in Matlab way, which sounds really cool
def port_data_to_yaml(DS_name, ds_gmm, A_k, att, x0_all, dt):
    Mu, Priors, Sigma, _ = load_learned_data_from_ML()
    new_Sig = np.copy(Sigma)
    new_A_k = np.copy(A_k)
    for k in np.arange(len(Sigma)):
        new_Sig[k] = new_Sig[k].T
        new_A_k[k] = new_A_k[k].T

    dt = float(dt)
    M = len(Mu)
    K = len(Mu[0])
    Sigma_in_ml = Sigma.reshape(K * M ** 2).tolist()
    Mu_in_ml = Mu.T.reshape(M * K).tolist()
    x0_all_in_ml = x0_all.T.reshape(M * len(x0_all[0])).tolist()
    att_in_ml = att.reshape(M).tolist()
    A_k_in_ml = new_A_k.reshape(K * M ** 2).tolist()
    Priors = np.copy(Priors).tolist()
    dic = {'name': DS_name, 'K': K, 'M': M, 'Priors': Priors, 'Mu': Mu_in_ml, 'Sigma': Sigma_in_ml, 'A': A_k_in_ml, 'attractor': att_in_ml, 'x0_all': x0_all_in_ml, 'dt': dt}
    stream = open('haruhi.yml', mode='w')
    print(yaml.dump(dic, stream))
