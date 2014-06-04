import theano as th
import theano.tensor as T
import numpy as np

import Optimizer as opt
import NeuralNet as NN
import autoencoder
import mlp

from utils import tile_raster_images
import PIL.Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cPickle as Pickle

def mnist_image(self, fname, weights):
    image = PIL.Image.fromarray(
        tile_raster_images( X=weights,
                            img_shape=(28, 28), tile_shape=(10, 10),
                            tile_spacing=(1, 1) ))
    
    image.save(fname)

if __name__ == '__main__':
    (x_train,y_train), (x_valid,y_valid), (x_test, y_test) = Pickle.load(open('mnist.pkl'))

    # ---- Define experimental architectures -----
    # architectures = [
    #     [NN.LayerData(40)],
    #     [NN.LayerData(200)],
    #     [NN.LayerData(200), NN.LayerData(200)],
    #     [NN.LayerData(500)],
    #     [NN.LayerData(500), NN.LayerData(50)],
    #     [NN.LayerData(397)]
    #     [NN.LayerData(397), NN.LayerData(203)]]

    # ---- Choose experimental learning rates -----
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes    = [10000, 1000, 100, 10]
    n_epochs       = 100

    params = [(learning_rate, batch_size) for learning_rate in learning_rates for batch_size in batch_sizes]
    results = {}
    for i,(learning_rate, batch_size) in enumerate(params):
        mlp500 = mlp.MLP(n_in = 784,
                            n_out = 10,
                            layers = [NN.LayerData(200)])
        print 'Training %d/%d parameter sets' %(i, len(params))
        results[(learning_rate, batch_size)] = opt.gradient_descent(mlp500,
                                                                    x_train = x_train,
                                                                    y_train = y_train,
                                                                    learning_rate = learning_rate,
                                                                    batch_size = batch_size,
                                                                    n_epochs = n_epochs,
                                                                    x_valid = x_valid,
                                                                    y_valid = y_valid)
    
    Pickle.dump(results, open('results_200.pkl','wb'))

    # Compare training across different learning rates and batch sizes
  


