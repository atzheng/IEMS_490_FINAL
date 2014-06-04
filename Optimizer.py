import numpy as np
import theano as th
import theano.tensor as T

# ----- COST FUNCTIONS -----

def negative_log_likelihood(output, y):
    return -T.mean(T.log(output)[T.arange(y.shape[0]), y])

def reconstruction_xentropy(output, x):
    return T.mean( - T.sum( x * T.log( output ) + (1 - x) * T.log( 1 - output ), axis = 1))

# ----- OPTIMIZERS -----
        
         
def gradient_descent(NN,
                     x_train,
                     y_train,
                    learning_rate=0.01,
                    n_epochs=1000,
                    batch_size=20,
                    x_valid = None,
                    y_valid = None):

    # Load data
    x_shared = th.shared(np.asarray(x_train, dtype = th.config.floatX),
                           borrow = True)
    if x_train is y_train:
        y_shared = x_shared
    else:
        y_shared = T.cast(th.shared(np.asarray(y_train, dtype = th.config.floatX),
                                  borrow = True),
                    'int32')

    # Initialize gradient descent function
    N = x_shared.get_value(borrow=True).shape[0]
    n_train_batches =  N / batch_size
    index = T.lscalar() 

    gparams = [T.grad(NN.loss,param) for param in NN.params]
    updates = [(param, param - learning_rate * gparam) for param,gparam in zip(NN.params,gparams)]

    backpropagate = th.function(inputs=[index], outputs=NN.loss,
            updates=updates,
            givens={
                NN.x: x_shared[index * batch_size:(index + 1) * batch_size],
                NN.y: y_shared[index * batch_size:(index + 1) * batch_size]})

    # Backpropagate
    loss = 0
    loss_history = []
    valid_history = []
    valid_freq = 1000
    
    for epoch in range(n_epochs):
        if (x_valid is not None
            and y_valid is not None):
            valid_error = NN.validation_error(x_valid, y_valid)
            print('Epoch %i: Validation Error %f %%'
                    % (epoch, valid_error))
            print('Epoch %i: Loss %f' %(epoch, loss))
            loss_history.append(loss)
            valid_history.append(valid_error)

        for minibatch_index in xrange(n_train_batches):
            loss = backpropagate(minibatch_index)
                
    return (valid_history, loss_history)




