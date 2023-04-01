"""
Implementation of ZINBAE model
"""

from time import time
import numpy as np
from keras.models import Model
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, GaussianNoise, Layer, Activation, Lambda, Multiply, BatchNormalization, Reshape, Concatenate, RepeatVector, Permute
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping

from sklearn.cluster import KMeans
from sklearn import metrics

import h5py
import scanpy.api as sc
from layers import ConstantDispersionLayer, SliceLayer, ColWiseMultLayer
from loss import poisson_loss, NB, ZINB, mse_loss_v2
from preprocess import read_dataset, normalize
import tensorflow as tf

from numpy.random import seed
seed(2211)
from tensorflow import set_random_seed
set_random_seed(2211)

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

def mean_MSE(x_impute, x_real):
    return np.mean(np.square(np.log(x_impute+1)-np.log(x_real+1)))

def imputate_error(x_impute, x_real, x_raw):
    x_impute_log = np.log(x_impute[(x_raw-x_real)<0]+1)
    x_real_log = np.log(x_real[(x_raw-x_real)<0]+1)
    return np.sum(np.abs(x_impute_log-x_real_log))/np.sum(x_real_log>0)

def autoencoder(dims, noise_sd=0, init='glorot_uniform', act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    sf_layer = Input(shape=(1,), name='size_factors')
    x = Input(shape=(dims[0],), name='counts')
    h = x
    h = GaussianNoise(noise_sd, name='input_noise')(h)
 
    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], kernel_initializer=init, name='encoder_%d' % i)(h)
        h = BatchNormalization(center=True, scale=False, name='encoder_batchnorm_%d' % i)(h)
        h = Activation(act, name='encoder_act_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_hidden')(h)  # hidden layer, features are extracted from here
    h = BatchNormalization(center=True, scale=False, name='encoder_hidden_batchnorm_%d' % i)(h)
    h = Activation(act, name='encoder_hidden_act')(h)

    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        h = Dense(dims[i], kernel_initializer=init, name='decoder_%d' % i)(h)
        h = BatchNormalization(center=True, scale=False, name='decoder_batchnorm_%d' % i)(h)
        h = Activation(act, name='decoder_act_%d' % i)(h)
    # output
 
    pi = Dense(dims[0], activation='sigmoid', kernel_initializer=init, name='pi')(h)

    disp = Dense(dims[0], activation=DispAct, kernel_initializer=init, name='dispersion')(h)

    mean = Dense(dims[0], activation=MeanAct, kernel_initializer=init, name='mean')(h)

    output = ColWiseMultLayer(name='output')([mean, sf_layer])
    output = SliceLayer(0, name='slice')([output, disp, pi])

    return Model(inputs=[x, sf_layer], outputs=output)

### Gumbel-softmax layer ###
def sampling_gumbel(shape, eps=1e-8):
    u = tf.random_uniform(shape, minval=0., maxval=1)
    return -tf.log(-tf.log(u+eps)+eps)

def compute_softmax(logits,temp):
    z = logits + sampling_gumbel( K.shape(logits) )
    return K.softmax( z / temp )

def gumbel_softmax(args):
    logits,temp = args
    y = compute_softmax(logits,temp)
    return y


class ZINB_AE(object):
    def __init__(self,
                 dims,
                 noise_sd=0,
                 ridge=0,
                 debug=False,
                 eps = 1e-20):
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.noise_sd = noise_sd
        self.act = 'relu'
        self.ridge = ridge
        self.debug = debug
        self.eps = eps

        self.autoencoder = autoencoder(self.dims, noise_sd=self.noise_sd, act = self.act)

        pi = self.autoencoder.get_layer(name='pi').output
        disp = self.autoencoder.get_layer(name='dispersion').output
        zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)
        self.zinb_loss = zinb.loss 

        # zero-inflated outputs
        tau_input = Input(shape=(self.dims[0],), name='tau_input')
        pi_ = self.autoencoder.get_layer('pi').output
        mean_ = self.autoencoder.output

        pi_log_ = Lambda(lambda x:tf.log(x+self.eps))(pi_)
        nondrop_pi_log_ = Lambda(lambda x:tf.log(1-x+self.eps))(pi_)
        pi_log_ = Reshape( target_shape=(self.dims[0],1) )(pi_log_)
        nondrop_pi_log_ = Reshape( target_shape=(self.dims[0],1) )(nondrop_pi_log_)     
        logits = Concatenate(axis=-1)([pi_log_,nondrop_pi_log_])
        temp_ = RepeatVector( 2 )(tau_input)
        temp_ = Permute( (2,1) )(temp_)
        samples_ = Lambda( gumbel_softmax,output_shape=(self.dims[0],2,) )( [logits,temp_] )          
        samples_ = Lambda( lambda x:x[:,:,1] )(samples_)
        samples_ = Reshape( target_shape=(self.dims[0],) )(samples_)      
        output_ = Multiply(name='ZI_output')([mean_, samples_])

        self.model = Model(inputs=[self.autoencoder.input[0], self.autoencoder.input[1], tau_input],
                           outputs=[output_, self.autoencoder.output])

    def pretrain(self, x, x_count, batch_size=256, epochs=200, optimizer='adam', ae_file='ae_weights.h5'):
        print('...Pretraining autoencoder...')
        self.autoencoder.compile(loss=self.zinb_loss, optimizer=optimizer)
        es = EarlyStopping(monitor="loss", patience=50, verbose=1)
        self.autoencoder.fit(x=x, y=x_count, batch_size=batch_size, epochs=epochs, callbacks=[es], shuffle=True)
        self.autoencoder.save_weights(ae_file)
        print('Pretrained weights are saved to ./' + str(ae_file))
        self.pretrained = True
    
    def fit(self, x, x_count, batch_size=256, maxiter=2e3, ae_weights=None, 
                loss_weights=[0.01, 1], optimizer='adam', model_file='model_weight.h5'):
        self.model.compile(loss={'ZI_output': mse_loss_v2, 'slice': self.zinb_loss}, loss_weights=loss_weights, optimizer=optimizer)
        if not self.pretrained and ae_weights is None:
            print('...pretraining autoencoders using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, x_count, batch_size)
            self.pretrained = True
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('ae_weights is loaded successfully.')

        # anneal tau
        tau0 = 1.
        min_tau = 0.5
        anneal_rate = 0.0003
        tau = tau0
#        es = EarlyStopping(monitor="loss", patience=20, verbose=1)
        for e in range(maxiter):
            if e % 100 == 0:
                tau = max( tau0*np.exp( -anneal_rate * e),min_tau   )
                tau_in = np.ones( x[0].shape,dtype='float32' ) * tau
                print(tau)
            print("Epoch %d/%d" % (e, maxiter))
            self.model.fit(x=[x[0], x[1], tau_in], y=x_count, batch_size=batch_size, epochs=1, shuffle=True)
        self.model.save_weights(model_file)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--max_iters', default=2000, type=int)
    parser.add_argument('--gamma', default=.01, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--ae_weight_file', default='ae_weights.h5')
    parser.add_argument('--model_weight_file', default='model_weights.h5')

    args = parser.parse_args()

    # load dataset
    optimizer = Adam(amsgrad=True)

    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    true_count = np.array(data_mat['true_count'])
    data_mat.close()

    x = np.floor(x)

    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    input_size = adata.n_vars

    print(adata.X.shape)
    print(y.shape)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)


    print(args)

    zinbae_model = ZINB_AE(dims=[input_size, 64, 32], noise_sd=2.5)
    zinbae_model.autoencoder.summary()
    zinbae_model.model.summary()

    # Pretrain autoencoders before clustering
    if args.ae_weights is None:
        zinbae_model.pretrain(x=[adata.X, adata.obs.size_factors], x_count=adata.raw.X, batch_size=args.batch_size, epochs=args.pretrain_epochs, 
                        optimizer=optimizer, ae_file=args.ae_weight_file)

    zinbae_model.fit(x=[adata.X, adata.obs.size_factors], x_count=[adata.raw.X, adata.raw.X], batch_size=args.batch_size, ae_weights=args.ae_weights,
                    maxiter=args.max_iters, loss_weights=[args.gamma, 1], optimizer=optimizer, model_file=args.model_weight_file)
    # Impute error
    x_impute = zinbae_model.autoencoder.predict(x=[adata.X, adata.obs.size_factors])

    raw_error = imputate_error(adata.raw.X, true_count, x_raw=adata.raw.X)
    imputation_error = imputate_error(x_impute, true_count, x_raw=adata.raw.X)
    print("Before imputation error: %.4f, after imputation error: %.4f" % (raw_error, imputation_error))
