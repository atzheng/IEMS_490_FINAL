import numpy as np
import theano as th
import theano.tensor as T

import NeuralNet as NN
import Optimizer as opt

class MLP(NN.NeuralNet):
    '''
    Multi-layer perceptron class. This is a special type of NeuralNet with:
    - A softmax output layer
    - A negative log likelihood loss function
    '''
    def __init__(self,
                 n_in,
                 n_out,
                 layers,
                 L1_reg = 0.,
                 L2_reg = 0.,
                 input = None
                 rng = np.random.RandomState(1234)):

        output_layer_args = NN.LayerData(
                            n_out = n_out,
                            activation = T.nnet.softmax)

        NN.NeuralNet.__init__(self,
                           n_in = n_in,
                           n_out = n_out,
                           layers = layers,
                           output_layer_args = output_layer_args,
                           error_fn = opt.negative_log_likelihood,
                           L1_reg = L1_reg,
                           L2_reg = L2_reg,
                           input = input,
                           rng = rng)
        
        self.predict = th.function([self.x],
                                       T.argmax( self.output_layer.output , axis = 1),
                                       allow_input_downcast = True)
        
    def validation_error(self, x_valid, y_valid):
        return 100*np.mean(self.predict(x_valid) != y_valid)

if __name__ == '__main__':
    from NeuralNet import LayerData
    import cPickle as Pickle
    import time
        
    learning_rate = 0.01
    L1_reg = 0.00
    L2_reg = 0.0001
    n_epochs = 2
    dataset = 'mnist.pkl.gz'
    batch_size = 20
    n_hidden = 500
    n_out = 10

    (xtrain,ytrain), (xvalid,yvalid), (xtest, ytest) = Pickle.load(open('mnist.pkl'))
    print '... building the model'

    classifier = MLP( n_in=28 * 28,
                      n_out=10,
                      layers = [LayerData(n_out=n_hidden, activation=T.tanh)],
                      L1_reg = L1_reg,
                      L2_reg = L2_reg)

    print '... training'
    start = time.clock()
    opt.gradient_descent(NN = classifier,
                        x_train = xtrain,
                         y_train = ytrain,
                         learning_rate = learning_rate,
                         n_epochs = n_epochs,
                         batch_size = batch_size,
                         x_valid = xvalid,
                         y_valid = yvalid)

