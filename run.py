import cPickle as Pickle
import numpy as np
import theano as th
import theano.tensor as T
import time
import NeuralNet

(xtrain,ytrain), (xvalid,yvalid), (xtest, ytest) = Pickle.load(open('mnist.pkl'))
init_wts = Pickle.load(open('initial_conditions.pkl'))

xtoy = xtrain[:2000,:]
ytoy = ytrain[:2000]

torun = {'logit' : True,
         'mlp_1' : False,
         'mlp_2' : False}


# Logit
if torun['logit']:
    print ' ----- LOGISTIC REGRESSION ----- '
    mnist_MLP = NeuralNet.MLP(xtrain,ytrain, layers = [
                                                    (10, T.nnet.softmax, np.zeros((10,784),dtype = th.config.floatX), None)])

    start = time.clock()
    mnist_MLP.train(epochs = 1000,
                    step_size = 0.13,
                    batch_size = 20,
                    X_valid = xvalid,
                    Y_valid = yvalid)
    preds = mnist_MLP.predict(xtest)
    print 'Training complete. Time elapsed: %.2f' %(time.clock() - start)
    print 'Error rate: %.2f' % (1-(float(sum(np.equal(preds,ytest)))/float(len(preds))))

# MLP
if torun['mlp_1']:
    print ' ----- MLP: 784 - 500 - 10 ----- '
    mnist_MLP = NeuralNet.MLP(xtrain,
                              ytrain,
                              layers = [(500, T.tanh, init_wts[0], None),
                                        (10, T.nnet.softmax, np.zeros((10,500), dtype = th.config.floatX), None)])

    start = time.clock()
    mnist_MLP.train(epochs = 100,
                    step_size = 0.01,
                    batch_size = 20,
                    L2_lambda = 0.0001,
                    X_valid = xvalid,
                    Y_valid = yvalid)
    
    preds = mnist_MLP.predict(xtest)
    print 'Training complete. Time elapsed: %.2f' %(time.clock() - start)
    print 'Error rate: %.2f' % (1-(float(sum(np.equal(preds,ytest)))/float(len(preds))))

if torun['mlp_2']:
    print ' ----- MLP: 784 - 500 - 10 - 10 ----- '
    mnist_MLP = NeuralNet.MLP(xvalid, yvalid,
                              layers = [(500, T.tanh, None, None),
                                        (10, T.tanh, None, None),
                                        (10, T.nnet.softmax, None, None)])

    start = time.clock()
    mnist_MLP.train(epochs = 300, step_size = 0.15, batch_size = 250)
    preds = mnist_MLP.predict(xtest)
    print 'Training complete. Time elapsed: %.2f' %(time.clock() - start)
    print 'Error rate: %.2f' % (1-(float(sum(np.equal(preds,ytest)))/float(len(preds))))




