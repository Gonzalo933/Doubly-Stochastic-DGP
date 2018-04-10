import matplotlib
import sys
import os

if not ("DISPLAY" in os.environ):  # Execution in the CCC
    matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
from sacred import Experiment
import numpy as np
import tensorflow as tf

from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.models.gpr import GPR
from gpflow.training import AdamOptimizer, ScipyOptimizer
from doubly_stochastic_dgp.dgp import DGP
from scipy.cluster.vq import kmeans2



def f1(x):
    return (x-0.5)**3 + np.sin(5*x) + 0.01*np.random.randn(x.shape[0], 1)

def f2(x):
    return (x-0.5)**3 + np.sin(5*x) + 0.1*np.random.randn(x.shape[0], 1)

def f3(x):
    return np.sin(5*x) + 0.05*np.random.randn(x.shape[0], 1)

def f4(x):
    return x**3 + 0.05*np.random.randn(x.shape[0], 1)

def f5(x):
    y = x.copy()
    y[y < 0.0] = 0.0
    y[y > 0.0] = 1.0
    return y + 0.05*np.random.randn(x.shape[0], 1)

def make_DGP(L, D_problem, D_hidden, X, Y, Z):
    kernels = []
    # First layer
    kernels.append(RBF(D_problem, lengthscales=0.2, variance=1.) + White(D_problem, variance=1e-5))
    for l in range(L-1):
        k = RBF(D_hidden, lengthscales=0.2, variance=1.) + White(D_hidden, variance=1e-5)
        kernels.append(k)

    m_dgp = DGP(X, Y, Z, kernels, Gaussian(), num_samples=10)

    # init the layers to near determinisic
    for layer in m_dgp.layers[:-1]:
        layer.q_sqrt = layer.q_sqrt.value * 1e-5
    return m_dgp


funcnames = {1: 'mixed_clean',
             2: 'mixed_noisy',
             3: 'sine',
             4: 'cubic',
             5: 'step'}

notrain = 500
notest = 200

X_train = np.reshape(np.linspace(-1, 1, notrain), (notrain, 1))
X_test = np.reshape(np.linspace(-1, 1, notest), (notest, 1))


print("N train: {}".format(notrain))
print("N test: {}".format(notest))
print("X_train: {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
#print("X_plot: {}".format(X_plot.shape))

n_samples = 100
max_iter = 500
M = 25
Z = np.random.uniform(0, 1, M)[:, None]
folder = 'figs'

L = 2
# 2 Layer, 1 and 2 nodes
for n_node in [1, 2]:
    for func in [1, 2, 3, 4, 5]:
        if func == 1:
            y_train = f1(X_train)
            y_test = f1(X_test)
        elif func == 2:
            y_train = f2(X_train)
            y_test = f2(X_test)
        elif func == 3:
            y_train = f3(X_train)
            y_test = f3(X_test)
        elif func == 4:
            y_train = f4(X_train)
            y_test = f4(X_test)
        elif func == 5:
            y_train = f5(X_train)
            y_test = f5(X_test)

        y_train = np.reshape(y_train, (notrain, 1))
        y_test = np.reshape(y_test, (notest, 1))

        m_dgp_2 = make_DGP(2, X_train.shape[-1], n_node, X_train, y_train, Z)

        AdamOptimizer(0.01).minimize(m_dgp_2, maxiter=max_iter)

        m, v = m_dgp_2.predict_y(X_test, n_samples)
        m = np.mean(m, 0)  # ?
        v = np.mean(v, 0)
        rmse = np.sqrt(np.mean((m - y_test) ** 2))

        plt.figure(num=None, figsize=(9, 6))
        sns.set()  # seaborn style
        plt.text(0.01, 0.85,
                 "RMSE: {:.4f}, LL: {:.4f}\nSamp(pred): {}, iter: {}\nLayers: {} M: {}\nfunc: {}".format(
                     rmse,
                     0.0,
                     n_samples,
                     max_iter, L,
                     M, func),
                 transform=plt.gcf().transFigure)

        plt.plot(X_train, y_train, 'bo', alpha=0.5)
        plt.plot(X_test, m, 'm-')
        plt.plot(X_test, m - 2 * np.sqrt(v), 'm--')
        plt.plot(X_test, m + 2 * np.sqrt(v), 'm--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.subplots_adjust(left=0.26)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f"{folder}/f{func}_L{L}_nnodes{n_node}.png")
        plt.close()
        print("Finish")


L = 3
# 3 Layer, 1 and 2 nodes
for n_node in [1, 2]:
    for func in [1, 2, 3, 4, 5]:
        if func == 1:
            y_train = f1(X_train)
            y_test = f1(X_test)
        elif func == 2:
            y_train = f2(X_train)
            y_test = f2(X_test)
        elif func == 3:
            y_train = f3(X_train)
            y_test = f3(X_test)
        elif func == 4:
            y_train = f4(X_train)
            y_test = f4(X_test)
        elif func == 5:
            y_train = f5(X_train)
            y_test = f5(X_test)

        y_train = np.reshape(y_train, (notrain, 1))
        y_test = np.reshape(y_test, (notest, 1))

        m_dgp_2 = make_DGP(2, X_train.shape[-1], n_node, X_train, y_train, Z)

        AdamOptimizer(0.01).minimize(m_dgp_2, maxiter=max_iter)

        m, v = m_dgp_2.predict_y(X_test, n_samples)
        m = np.mean(m, 0)  # ?
        v = np.mean(v, 0)
        rmse = np.sqrt(np.mean((m - y_test) ** 2))

        plt.figure(num=None, figsize=(9, 6))
        sns.set()  # seaborn style
        plt.text(0.01, 0.85,
                 "RMSE: {:.4f}, LL: {:.4f}\nSamp(pred): {}, iter: {}\nLayers: {} M: {}\nfunc: {}".format(
                     rmse,
                     0.0,
                     n_samples,
                     max_iter, L,
                     M, func),
                 transform=plt.gcf().transFigure)

        plt.plot(X_train, y_train, 'bo', alpha=0.5)
        plt.plot(X_test, m, 'm-')
        plt.plot(X_test, m - 2 * np.sqrt(v), 'm--')
        plt.plot(X_test, m + 2 * np.sqrt(v), 'm--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.subplots_adjust(left=0.26)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f"{folder}/f{func}_L{L}_nnodes{n_node}.png")
        plt.close()
        print("Finish")

L = 4
# 4 Layer, 1 and 2 nodes
for n_node in [1, 2]:
    for func in [1, 2, 3, 4, 5]:
        if func == 1:
            y_train = f1(X_train)
            y_test = f1(X_test)
        elif func == 2:
            y_train = f2(X_train)
            y_test = f2(X_test)
        elif func == 3:
            y_train = f3(X_train)
            y_test = f3(X_test)
        elif func == 4:
            y_train = f4(X_train)
            y_test = f4(X_test)
        elif func == 5:
            y_train = f5(X_train)
            y_test = f5(X_test)

        y_train = np.reshape(y_train, (notrain, 1))
        y_test = np.reshape(y_test, (notest, 1))

        m_dgp_2 = make_DGP(2, X_train.shape[-1], n_node, X_train, y_train, Z)

        AdamOptimizer(0.01).minimize(m_dgp_2, maxiter=max_iter)

        m, v = m_dgp_2.predict_y(X_test, n_samples)
        m = np.mean(m, 0)  # ?
        v = np.mean(v, 0)
        rmse = np.sqrt(np.mean((m - y_test) ** 2))

        plt.figure(num=None, figsize=(9, 6))
        sns.set()  # seaborn style
        plt.text(0.01, 0.85,
                 "RMSE: {:.4f}, LL: {:.4f}\nSamp(pred): {}, iter: {}\nLayers: {} M: {}\nfunc: {}".format(
                     rmse,
                     0.0,
                     n_samples,
                     max_iter, L,
                     M, func),
                 transform=plt.gcf().transFigure)

        plt.plot(X_train, y_train, 'bo', alpha=0.5)
        plt.plot(X_test, m, 'm-')
        plt.plot(X_test, m - 2 * np.sqrt(v), 'm--')
        plt.plot(X_test, m + 2 * np.sqrt(v), 'm--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.subplots_adjust(left=0.26)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f"{folder}/f{func}_L{L}_nnodes{n_node}.png")
        plt.close()
        print("Finish")