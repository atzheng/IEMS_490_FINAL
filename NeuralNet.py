import theano as th
import theano.tensor as T
from random_sampler import shared_sampler
import numpy as np
import math

class HiddenLayer:
    
    def __init__(self, inputs, n_inputs, n_outputs,
                 activation = T.tanh,
                 w_init = None,
                 b_init = None
                 ):

        if w_init is None:

            w_init = np.asarray(np.random.uniform( low=-np.sqrt(6. / (n_inputs + n_outputs)),
                                                        high=np.sqrt(6. / (n_inputs + n_outputs)),
                                                        size=(n_outputs, n_inputs)), dtype=th.config.floatX)

        else:
            assert w_init.shape == (n_outputs, n_inputs)
            
        if b_init is None:
            b_init = np.random.uniform( size = n_outputs )
        else:
            assert b_init.shape == (n_outputs,)
            
        self.w = th.shared( w_init, 'w')
        self.b = th.shared( b_init, 'b' )

        self.n_in = n_inputs
        self.n_out = n_outputs
        self.activation = activation
        self.inputs = inputs

        self.output = self.activation( T.dot(self.inputs, self.w.T) + self.b )

class MLP:
    def __init__(self, X, Y, layers):

        self.X = X
        self.Y = Y
        self.K = Y.shape[1]
        
        self.N,self.m = X.shape
        self.x = T.matrix( 'x' )
        self.y = T.matrix( 'y' )
        
        assert self.K >= 2
        assert self.N == Y.shape[0]
        
        self.params = []
        prev_output = self.x
        prev_size = self.m
        
        for size, activation, w_init, b_init in layers:
            H = HiddenLayer(prev_output, prev_size, size, activation, w_init, b_init)
            self.params += [H.w, H.b]
            prev_output = H.output
            prev_size = size

        self.output = prev_output

        # Functions
        self.predict = th.function([self.x], T.argmax( self.output , axis = 1 ), allow_input_downcast = True)
#        self.predict = th.function([self.x], self.output, allow_input_downcast = True)
        self.loss = - T.mean( self.output * self.y )

    def train(self, epochs = 100, step_size = 0.1, batch_size = 1):
        index = T.lscalar()
        grad = [T.grad(self.loss, param) for param in self.params]
        updates = [(param, param - step_size * grad) for param,grad in zip(self.params, grad)]
        
        sampler = shared_sampler(self.X, self.Y)

        refresh_iter = math.floor(self.N/batch_size)
        
        for epoch in range(epochs):
            if epoch % 10 == 0 : print 'Epoch %d' %epoch
            shared_X, shared_Y = sampler.get_order()
            SGD = th.function(inputs = [index],
                              outputs = [self.loss],
                              updates = updates,
                              givens = {self.x: shared_X[(index * batch_size): ((index + 1) * batch_size)],
                                        self.y: shared_Y[(index * batch_size): ((index + 1) * batch_size)]},
                              allow_input_downcast = True)

            for k in range(int(math.floor(self.N/batch_size))):
                loss = SGD(k)

def compact_2_hotone(Y):
    # Assumes that Y is a vector of max(Y) numbered classes
    Y_hotone = np.zeros(shape = (len(Y), max(Y) + 1))
    Y_hotone[range(len(Y)),Y] = 1
    return Y_hotone
            
if __name__ == '__main__':
    import time
    # Test the xor gate
    X = np.repeat(np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]]), 250, 0)
    Y = compact_2_hotone(np.repeat(np.array([0, 1, 1, 0]),250))

    w_init_HL = np.array([[-.2, .2], [.5, -1.]])
    b_init_HL = np.array([0., .2])
    
    w_init_O = np.array([[.1,1.],[-.1,-1.]])

    b_init_O = np.array([-.1, -.1])
    
    # xor_NN = MLP(X, Y, layers = [(2, T.nnet.sigmoid, w_init_HL, b_init_HL),
    #                              (2, T.nnet.softmax, w_init_O, b_init_O)])
    xor_NN = MLP(X, Y, layers = [(2, T.tanh, None, None),
                                 (2, T.nnet.softmax, None, None)])

    start = time.clock()
    xor_NN.train(epochs = 100, batch_size = 1)
    print 'Training complete. Time elapsed: %.2f' %(time.clock() - start)

    X_test = np.array([ [0,0],
                        [0,1],
                        [1,0],
                        [1,1] ])
    
    pred = xor_NN.predict(X_test)
    print 'Predictions:'
    print pred
        
    
    
