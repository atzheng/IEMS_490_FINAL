import theano as th
import theano.tensor as T
import numpy as np

import NeuralNet as NN
import Optimizer as opt

class AutoEncoder(NN.NeuralNet):
    def __init__(self,
                 n_in,
                 layers,
                 rng = np.random.RandomState(1234)):

        output_layer_args = NN.LayerData(n_out = n_in, activation = T.nnet.sigmoid)
        NN.NeuralNet.__init__(self,
                              n_in,
                              n_in,
                              layers,
                              output_layer_args = output_layer_args,
                              error_fn = opt.reconstruction_xentropy
                              rng)


        
if __name__ == '__main__':
    AE = AutoEncoder(n_in = 784,
                     layers = [NN.LayerData(n_out = 500)])
                                                   

                                
                                
                                
            
