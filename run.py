import cPickle as Pickle
import numpy as np
import theano.tensor as T
import time
import NeuralNet

(xtrain,ytrain), (xvalid,yvalid), (xtest, ytest) = Pickle.load(open('mnist.pkl'))

xtoy = xtrain[:2000,:]
ytoy = ytrain[:2000]

# Logit
# print ' ----- LOGISTIC REGRESSION ----- '
# mnist_MLP = NeuralNet.MLP(xtrain, NeuralNet.compact_2_hotone(ytrain), layers = [
#                                                 (10, T.nnet.softmax, None, None)])

# start = time.clock()
# mnist_MLP.train(epochs = 100, step_size = 0.13, batch_size = 600)
# preds = mnist_MLP.predict(xtest)
# print 'Training complete. Time elapsed: %.2f' %(time.clock() - start)
# print 'Error rate: %.2f' % (1-(float(sum(np.equal(preds,ytest)))/float(len(preds))))

# MLP
print ' ----- MLP: 784 - 500 - 10 ----- '
mnist_MLP = NeuralNet.MLP(xtrain, NeuralNet.compact_2_hotone(ytrain), layers = [(500, T.tanh, None, None),
                                                (10, T.nnet.softmax, None, None)])

start = time.clock()
mnist_MLP.train(epochs = 100, step_size = 0.15, batch_size = 250)
preds = mnist_MLP.predict(xtest)
print 'Training complete. Time elapsed: %.2f' %(time.clock() - start)
print 'Error rate: %.2f' % (1-(float(sum(np.equal(preds,ytest)))/float(len(preds))))





