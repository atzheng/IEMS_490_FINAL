import numpy as np
import theano as th
import theano.tensor as T

class LayerData:
    def __init__(self, n_out, W= None, b= None, activation = T.tanh):
        self.n_out = n_out
        self.W = W
        self.b = b
        self.activation = activation

class HiddenLayer(object):
    def __init__(self,input, rng, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input

        self.W = self.init_W(W,n_in,n_out,rng,activation)
        self.b = self.init_b(b,n_out)

        self.output = self.determine_output(input,activation)

        self.y_pred = T.argmax(self.output, axis=1)

        self.params = [self.W, self.b]

    def init_W(self, W, n_in, n_out, rng, activation):
        if W is None:
            random_arr = rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out))
            W_values = np.asarray(random_arr, dtype=th.config.floatX)
            if activation == th.tensor.nnet.sigmoid:
                W_values *= 4
        else:
            W_values = W
        W = th.shared(value=W_values, name='W', borrow=True)
        return W

    def init_b(self, b, n_out):
        if b is None:
            b_values = np.zeros((n_out,), dtype=th.config.floatX)
        else:
            b_values = b
        b = th.shared(value=b_values, name='b', borrow=True)
        return b

    def determine_output(self,input,activation):
        lin_output = T.dot(input, self.W) + self.b
        return (lin_output if activation is None else activation(lin_output))

class NeuralNet:
    def __init__(self,
                 n_in,
                 n_out,
                 layers,
                 output_layer_args,
                 error_fn,
                 L1_reg = 0.,
                 L2_reg = 0.,
                 input = None,
                 output = None,
                 rng = np.random.RandomState(1234)):

        # Initialize variables
        self.x = (input if input is not None
                  else T.matrix('x'))
        self.y = (output if output is not None
                  else T.ivector('y'))
        self.rng = rng
        self.init_layers(layers, n_in, n_out, output_layer_args)

        # Initialize the optimizer
        self.loss = error_fn(self.output_layer.output, self.y)
        if L1_reg != 0:
            self.loss += L1_reg * self.L1
        if L2_reg != 0:
            self.loss += L2_reg * self.L2_sqr
            
        self.predict = None

    def init_layers(self, layers, n_in, n_out, output_layer_args):
        L1 = 0
        L2_sqr = 0

        self.params = []
        prev_output = self.x
        prev_size = n_in
        
        for layer_arg in layers:
            H = HiddenLayer(input = prev_output,
                            rng = self.rng,
                            n_in = prev_size,
                            n_out = layer_arg.n_out,
                            W = layer_arg.W,
                            b = layer_arg.b,
                            activation = layer_arg.activation)

            self.params += [H.W,H.b]
            L1 += T.sum(abs(H.W))
            L2_sqr += T.sum(H.W ** 2)
            prev_output = H.output
            prev_size = layer_arg.n_out

        self.output_layer = HiddenLayer(
            input=prev_output,
            rng = self.rng,
            n_in= prev_size,
            n_out = output_layer_args.n_out,
            activation = output_layer_args.activation,
            W =  (output_layer_args.W if output_layer_args.W is not None
                  else np.zeros((prev_size,n_out), dtype=th.config.floatX)),
            b = output_layer_args.b)
        
        self.params += [self.output_layer.W, self.output_layer.b]
        self.L1 = L1 + T.sum( abs( self.output_layer.W ))
        self.L2_sqr = L2_sqr + T.sum( self.output_layer.W ** 2 )

    def validation_error(self, x_valid, y_valid):
        assert False

        
