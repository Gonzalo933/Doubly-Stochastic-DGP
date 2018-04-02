# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np

from gpflow.params import DataHolder, Minibatch
from gpflow import autoflow, params_as_tensors, ParamList
from gpflow.models.model import Model
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow import settings
float_type = settings.float_type

from doubly_stochastic_dgp.layers import SVGP_Layer
from doubly_stochastic_dgp.utils import BroadcastingLikelihood


class DGP_Base(Model):
    """
    The base class for Deep Gaussian process models.

    Implements a Monte-Carlo variational bound and convenience functions.

    """
    def __init__(self, X, Y, likelihood, layers,
                 minibatch_size=None,
                 num_samples=1):
        Model.__init__(self)
        self.num_samples = num_samples

        self.num_data = X.shape[0]
        if minibatch_size:
            self.X = Minibatch(X, minibatch_size, seed=0)
            self.Y = Minibatch(Y, minibatch_size, seed=0)
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)

        self.likelihood = BroadcastingLikelihood(likelihood)

        self.layers = ParamList(layers)

    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1, zs=None):
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])

        Fs, Fmeans, Fvars = [], [], []

        F = sX
        zs = zs or [None, ] * len(self.layers)
        for layer, z in zip(self.layers, zs):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    @params_as_tensors
    def _build_predict(self, X, full_cov=False, S=1):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S)
        return Fmeans[-1], Fvars[-1]

    def E_log_p_Y(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with MC samples
        """
        Fmean, Fvar = self._build_predict(X, full_cov=False, S=self.num_samples)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  # S, N, D
        return tf.reduce_mean(var_exp, 0)  # N, D

    @params_as_tensors
    def _build_likelihood(self):
        L = tf.reduce_sum(self.E_log_p_Y(self.X, self.Y))
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        scale = tf.cast(self.num_data, float_type)
        scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size
        return L * scale - KL

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=False, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f_full_cov(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=True, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers_full_cov(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=True, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)


class DGP(DGP_Base):
    """
    This is the Doubly-Stochastic Deep GP, with linear/identity mean functions at each layer.

    The key reference is

    ::
      @inproceedings{salimbeni2017doubly,
        title={Doubly Stochastic Variational Inference for Deep Gaussian Processes},
        author={Salimbeni, Hugh and Deisenroth, Marc},
        booktitle={NIPS},
        year={2017}
      }

    """
    def __init__(self, X, Y, Z, kernels, likelihood,
                 num_outputs=None,
                 mean_function=Zero(),  # the final layer mean function
                 **kwargs):
        Model.__init__(self)
        num_outputs = num_outputs or Y.shape[1]

        # init the layers
        layers = []

        # inner layers
        X_running, Z_running = X.copy(), Z.copy()
        for kern_in, kern_out in zip(kernels[:-1], kernels[1:]):
            dim_in = kern_in.input_dim
            dim_out = kern_out.input_dim

            mf = Zero()  # Added to compare with DGP EP MCM
            # Inducing points for layer
            if Z.shape[1] > dim_in:  # Reduce Z by doing PCA
                _, _, V = np.linalg.svd(X, full_matrices=False)  # V -> (D,D) Matrix
                Z_kern = Z.dot(V[:dim_in, :].T)
            elif Z.shape[1] < dim_in:  # Increase Z by doing tile
                _, _, V = np.linalg.svd(X, full_matrices=False)  # V -> (D,D) Matrix
                first_pca = Z.dot(V[0, :].T)  # First Principal component
                Z_kern = np.tile(first_pca[:, None], (1, dim_in))
            else:  # same dimension
                Z_kern = Z.copy()
            layers.append(SVGP_Layer(kern_in, Z_kern, dim_out, mf))
            print("{} \n{}\n".format(Z_kern.shape, Z_kern[0:3, 0:3]))
        # Final Layer
        mf = Zero()  # Added to compare with DGP EP MCM
        dim_in = kernels[-1].input_dim
        # Inducing points for layer
        if Z.shape[1] > dim_in:  # Reduce Z by doing PCA
            _, _, V = np.linalg.svd(X, full_matrices=False)  # V -> (D,D) Matrix
            Z_kern = Z.dot(V[:dim_in, :].T)
        elif Z.shape[1] < dim_in:  # Increase Z by doing tile
            _, _, V = np.linalg.svd(X, full_matrices=False)  # V -> (D,D) Matrix
            first_pca = Z.dot(V[0, :].T)  # First Principal component
            Z_kern = np.tile(first_pca[:, None], (1, dim_in))
        else:  # same dimension
            Z_kern = Z.copy()
        print("{} \n{}\n".format(Z_kern.shape, Z_kern[0:3, 0:3]))
        layers.append(SVGP_Layer(kernels[-1], Z_kern, num_outputs, mean_function))

        """
        for kern_in, kern_out in zip(kernels[:-1], kernels[1:]):
            dim_in = kern_in.input_dim
            dim_out = kern_out.input_dim

            if dim_in == dim_out:
                mf = Identity()
            else:  # stepping down, use the pca projection
                _, _, V = np.linalg.svd(X_running, full_matrices=False)  # V -> (D,D) Matrix
                W = V[:dim_out, :].T

                mf = Linear(W)
                mf.set_trainable(False)
            # Z_kern = Z_running[:, 0:dim_in]
            # print("{} \n{}\n".format(Z_kern.shape, Z_kern[0:3, 0:3]))
            layers.append(SVGP_Layer(kern_in, Z_running, dim_out, mf))

            if dim_in != dim_out:
                Z_running = Z_running.dot(W)
                X_running = X_running.dot(W)
        """
        # final layer
        # Z_kern = Z_running[:, 0:kernels[-1].input_dim]
        # print("{} \n{}\n".format(Z_kern.shape, Z_kern[0:3, 0:3]))
        # layers.append(SVGP_Layer(kernels[-1], Z_running, num_outputs, mean_function))

        DGP_Base.__init__(self, X, Y, likelihood, layers, **kwargs)
