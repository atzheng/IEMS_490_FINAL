import cPickle as Pickle
import theano.tensor as T
import time
import NeuralNet

(xtrain,ytrain), (xvalid,yvalid), (xtest, ytest) = Pickle.load(open('mnist.pkl'))

xtoy = xtrain[:2000,:]
ytoy = ytrain[:2000]

mnist_MLP = NeuralNet.MLP(xtoy, ytoy, layers = [(500, T.tanh, None, None),
                                                (1, T.nnet.softmax, None, None)])

start = time.clock()
mnist_MLP.train(epochs = 100, step_size = 0.01)
preds = mnist_MLP.predict(xtest)
print 'Time elapsed: %.2f' %(time.clock() - start)




