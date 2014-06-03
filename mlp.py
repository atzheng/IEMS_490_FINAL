__docformat__ = 'restructedtext en'


import cPickle as Pickle
import gzip
import os
import sys
import time

import numpy

import theano
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

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))

        self.y_pred = T.argmax(self.output, axis=1)

        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, n_in, n_out, layers, rng = numpy.random.RandomState(1234)):

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        self.rng = rng
        self.init_layers(layers, n_in, n_out)

        self.negative_log_likelihood = -T.mean(T.log(self.output_layer.output)[T.arange(self.y.shape[0]), self.y])
        self.predict = theano.function([self.x],
                                       T.argmax( self.output_layer.output , axis = 1),
                                       allow_input_downcast = True)

    def init_layers(self, layers, n_in, n_out):

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
            n_out=n_out,
            activation = T.nnet.softmax,
            W = theano.shared(value=numpy.zeros((prev_size,n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True))
        
        self.L1 = L1 + T.sum( abs( self.output_layer.W ))
        self.L2_sqr = L2_sqr + T.sum( self.output_layer.W ** 2 )

        self.params += [self.output_layer.W, self.output_layer.b]

    def train(self,
              x,
              y,
              learning_rate=0.01,
              L1_reg=0.00,
              L2_reg=0.0000,
              n_epochs=1000,
              batch_size=20,
              xvalid = None,
              yvalid = None):

        xtrain = theano.shared(numpy.asarray(x, dtype = theano.config.floatX),
                               borrow = True)

        ytrain = T.cast(theano.shared(numpy.asarray(y, dtype = theano.config.floatX),
                                      borrow = True),
                        'int32')

        n_train_batches = xtrain.get_value(borrow=True).shape[0] / batch_size
        index = T.lscalar()

        cost = self.negative_log_likelihood + L1_reg * self.L1 + L2_reg * self.L2_sqr
         
        gparams = [T.grad(cost,param) for param in self.params]
        updates = [(param, param - learning_rate * gparam) for param,gparam in zip(self.params,gparams)]

        train_model = theano.function(inputs=[index], outputs=cost,
                updates=updates,
                givens={
                    self.x: xtrain[index * batch_size:(index + 1) * batch_size],
                    self.y: ytrain[index * batch_size:(index + 1) * batch_size]})

        for epoch in range(n_epochs):
            if xvalid is not None and yvalid is not None:
                print('Epoch %i: Validation Error %f %%'
                       % (epoch, 100*numpy.mean(self.predict(xvalid) != yvalid)))
            for minibatch_index in xrange(n_train_batches):
                loss = train_model(minibatch_index)

if __name__ == '__main__':

    learning_rate = 0.01
    L1_reg = 0.00
    L2_reg = 0.0001
    n_epochs = 70
    dataset = 'mnist.pkl.gz'
    batch_size = 20
    n_hidden = 500

    (xtrain,ytrain), (xvalid,yvalid), (xtest, ytest) = Pickle.load(open('mnist.pkl'))
    print '... building the model'

    classifier = MLP( n_in=28 * 28,
                      n_out=10,
                      layers = [LayerData(n_out=n_hidden, activation=T.tanh)])

    print '... training'
    start = time.clock()
    classifier.train(x = xtrain,
                     y = ytrain,
                     learning_rate = learning_rate,
                     L1_reg = L1_reg,
                     L2_reg = L2_reg,
                     n_epochs = n_epochs,
                     batch_size = batch_size,
                     xvalid = xvalid,
                     yvalid = yvalid)

    test_mlp()
