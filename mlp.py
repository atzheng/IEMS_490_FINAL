__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from logistic_sgd import load_data

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
        
        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])
    
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class MLP(object):
    def __init__(self, n_in, n_out, layers, rng = numpy.random.RandomState(1234)):

        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        self.rng = rng
        self.init_layers(layers, n_in, n_out)

        self.negative_log_likelihood = self.output_layer.negative_log_likelihood
        self.errors = self.output_layer.errors

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

    def train(self, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0000, n_epochs=1000,
                 dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
        datasets = load_data(dataset)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

        index = T.lscalar()  # index to a [mini]batch

        cost = self.negative_log_likelihood(self.y) \
         + L1_reg * self.L1 \
         + L2_reg * self.L2_sqr
         
        test_model = theano.function(inputs=[index],
                outputs=self.errors(self.y),
                givens={
                    self.x: test_set_x[index * batch_size:(index + 1) * batch_size],
                    self.y: test_set_y[index * batch_size:(index + 1) * batch_size]})

        validate_model = theano.function(inputs=[index],
                outputs=self.errors(self.y),
                givens={
                    self.x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                    self.y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)
        updates = []

        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        train_model = theano.function(inputs=[index], outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[index * batch_size:(index + 1) * batch_size],
                    self.y: train_set_y[index * batch_size:(index + 1) * batch_size]})

        validation_frequency = n_train_batches

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                         (epoch, minibatch_index + 1, n_train_batches,
                          this_validation_loss * 100.))

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
            dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    print '... building the model'

    classifier = MLP( n_in=28 * 28,
                      n_out=10,
                      layers = [LayerData(n_out=n_hidden, activation=T.tanh)])


    print '... training'
    classifier.train(learning_rate, L1_reg, L2_reg, n_epochs, dataset, batch_size, n_hidden)

if __name__ == '__main__':
    test_mlp()
