import theano as th
import theano.tensor as T
import numpy as np

import NeuralNet as NN
import Optimizer as opt

class AutoEncoder(NN.NeuralNet):
    def __init__(self,
                 n_in,
                 layers,
                 input = None,
                 rng = np.random.RandomState(1234)):

        x = input if input is not None else T.matrix('x')
            
        output_layer_args = NN.LayerData(n_out = n_in,
                                         activation = T.nnet.sigmoid)
                                         
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

class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self,
                 n_in,
                 layers,
                 corruption_level,
                 rng = np.random.RandomState(1234)):
        
        x = self.corrupt_data(T.matrix('x'), corruption_level, rng)
        AutoEncoder.__init__(self,
                             n_in,
                             layers,
                             input = x,
                             rng = rng)
        
    def corrupt_data(self, data, corruption_level, rng):
        theano_rng = T.shared_randomstreams.RandomStreams(rng.randint(2 ** 30))        
        return theano_rng.binomial(size=data.shape, n=1, p=1 - corruption_level, dtype = th.config.floatX) * data
    
if __name__ == '__main__':
    import cPickle as Pickle
    (xtrain,ytrain), (xvalid,yvalid), (xtest, ytest) = Pickle.load(open('mnist.pkl'))

    rng = np.random.RandomState()
    random_arr = rng.uniform(
        low=-np.sqrt(6. / (784 + 500)),
        high=np.sqrt(6. / (784 + 500)),
        size=(784, 500))
    
    W_init = th.shared(np.asarray(random_arr, dtype=th.config.floatX), 'W', borrow = True)

    # Autoencoder - NO NOISE
    AE = AutoEncoder(n_in = 784,
                     layers = [NN.LayerData(n_out = 500, W = W_init)],
                     rng = rng)

    print '... Training Autoencoder'
    opt.gradient_descent(AE,
                         x_train = xtrain,
                         y_train = xtrain,
                         learning_rate = 0.1,
                         batch_size = 20,
                         n_epochs = 20,
                         x_valid = xvalid,
                         y_valid = xvalid)
                         
    # Autoencoder - 30% NOISE
    DAE = DenoisingAutoEncoder(n_in = 784,
                                layers = [NN.LayerData(n_out = 500, W = W_init)],
                                corruption_level = 0.3,
                                rng = rng)


    print '... Training Autoencoder'
    opt.gradient_descent(DAE,
                         x_train = xtrain,
                         y_train = xtrain,
                         learning_rate = 0.1,
                         batch_size = 20,
                         n_epochs = 20,
                         x_valid = xvalid,
                         y_valid = xvalid)


                         
                                                   

                                
                                
                                
            
