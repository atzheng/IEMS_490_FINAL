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

        x = T.matrix('x')
        output_layer_args = NN.LayerData(n_out = n_in,
                                         activation = T.nnet.sigmoid,
                                         W = layers[-1].W) # Tied Weights
                                         
        NN.NeuralNet.__init__(self,
                              n_in,
                              n_in,
                              layers,
                              output_layer_args = output_layer_args,
                              error_fn = opt.reconstruction_xentropy,
                              input = x,
                              output = x,
                              rng = rng)
        
        self.reconstruct = th.function([self.x],
                                       self.output_layer.output)
                                   
    def validation_error(self, x_valid, y_valid):
        return np.mean( abs(self.reconstruct(x_valid) - x_valid) )

        
if __name__ == '__main__':
    import cPickle as Pickle
    (xtrain,ytrain), (xvalid,yvalid), (xtest, ytest) = Pickle.load(open('mnist.pkl'))
    AE = AutoEncoder(n_in = 784,
                     layers = [NN.LayerData(n_out = 500)],
                     rng = np.random.RandomState())

    print '... Training Autoencoder'
    opt.gradient_descent(AE,
                         x_train = xtrain,
                         y_train = xtrain,
                         learning_rate = 0.1,
                         batch_size = 20,
                         n_epochs = 15,
                         x_valid = xvalid,
                         y_valid = xvalid)
                         
                         
                         
                                                   

                                
                                
                                
            
