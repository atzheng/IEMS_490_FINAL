import theano as th
import theano.tensor as T
import numpy as np
import ggplot as gg
import math


class HiddenLayer:
    def __init__(self, inputs, n_inputs, n_outputs,
                 activation = T.tanh,
                 w_init = None,
                 b_init = None
                 ):

        if w_init is None:
            w_init = np.random.uniform( size = (n_outputs, n_inputs) )
        else:
            print w_init
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
        
        self.N,self.m = X.shape
        self.x = T.matrix( 'x' )
        self.y = T.vector( 'y' )
        
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
        self.predict = th.function([self.x], self.output)
        self.loss = T.mean(-self.y * T.log( self.output ) - ( 1 - self.y ) * T.log( 1 - self.output ))

    def train(self, epochs = 10000, step_size = 0.1, batch_size = 1):
        
        grad = [T.grad(self.loss, param) for param in self.params]
        updates = [(param, param - step_size * grad) for param,grad in zip(self.params, grad)]
        SGD = th.function(inputs = [self.x, self.y],
                       outputs = [self.loss],
                       updates = updates)
        
        num_iter = int(math.ceil(self.N*epochs/batch_size))
        for k in range(num_iter):
            idx = np.random.randint(0, self.N, batch_size)
            X_k = self.X[idx,:]
            Y_k = self.Y[idx]
            loss = SGD(X_k, Y_k)

if __name__ == '__main__':
    # Test the xor gate
    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])
    Y = np.array([0, 1, 1, 0])

    w_init_HL = np.array([[-.2, .2], [.5, -1.]])
    b_init_HL = np.array([0., .2])
    
    w_init_O = np.array([[.1,1.]])

    b_init_O = np.array([-.1])
    
    xor_NN = MLP(X, Y, layers = [(2, T.nnet.sigmoid, w_init_HL, b_init_HL),
                                 (1, T.nnet.sigmoid, w_init_O, b_init_O)])
    xor_NN.train()

    pred = xor_NN.predict(X)
    print pred
        
    
    
