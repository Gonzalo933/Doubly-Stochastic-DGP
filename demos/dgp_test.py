import numpy as np
import tensorflow as tf
import time

import matplotlib.pyplot as plt

from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.mean_functions import Constant
from gpflow.models.sgpr import SGPR, GPRFITC
from gpflow.models.svgp import SVGP
from gpflow.models.gpr import GPR
from gpflow.training import AdamOptimizer, ScipyOptimizer

from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp

from doubly_stochastic_dgp.dgp import DGP
from datasets import Datasets
datasets = Datasets(data_path='data/')

data = datasets.all_datasets['kin8nm'].get_data()
X, Y, Xs, Ys, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_std']]
print('N: {}, D: {}, Ns: {}'.format(X.shape[0], X.shape[1], Xs.shape[0]))

Z_100 = kmeans2(X, 100, minit='points')[0]
Z_5 = kmeans2(X, 5, minit='points')[0]
Z_500 = kmeans2(X, 500, minit='points')[0]


def make_dgp(X, Y, Z, L):
    D = X.shape[1]

    # the layer shapes are defined by the kernel dims, so here all hidden layers are D dimensional
    kernels = []
    #for l in range(L):
    kernels.append(RBF(5))
    kernels.append(RBF(2))
    kernels.append(RBF(9))

    # between layer noise (doesn't actually make much difference but we include it anyway)
    #for kernel in kernels[:-1]:
    #    kernel += White(D, variance=1e-5)

    mb = 1000 if X.shape[0] > 1000 else None
    model = DGP(X, Y, Z, kernels, Gaussian(), num_samples=10, minibatch_size=mb)

    # start the inner layers almost deterministically
    for layer in model.layers[:-1]:
        layer.q_sqrt = layer.q_sqrt.value * 1e-5

    return model


#m_dgp1 = make_dgp(X, Y, Z_100, 1)
#m_dgp2 = make_dgp(X, Y, Z_100, 2)
print("{} \n{}\n".format(Z_5.shape, Z_5[0:3, 0:3]))
print("-----------------")
m_dgp3 = make_dgp(X, Y, Z_5, 3)
#m_dgp4 = make_dgp(X, Y, Z_100, 4)
#m_dgp5 = make_dgp(X, Y, Z_100, 5)

AdamOptimizer(0.01).minimize(m_dgp3, maxiter=10)
print("Done")
