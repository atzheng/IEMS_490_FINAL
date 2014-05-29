import theano as th
import numpy as np

class shared_sampler:
    def __init__(self, X, Y):
        self.N = X.shape[0]
        assert self.N == len(Y)
        
        self.idx = 0
        self.X, self.Y = X, Y

    def get_order(self):
        self.reshuffle()
        return self.shared_X,self.shared_Y

    # def draw_sample(self, batch_size):
    #     assert batch_size <= self.N
    #     if self.N - self.idx < batch_size:
    #         self.reshuffle()
    #         self.idx = 0
    #     else:
    #         to_return =(self.shared_X[self.idx:self.idx+batch_size],
    #                     self.shared_Y[self.idx:self.idx+batch_size])
    #         self.idx += batch_size
    #         return to_return

    def reshuffle(self):
        self.order = np.random.permutation(range(self.N))
        self.X,self.Y = self.X[self.order], self.Y[self.order]
        
        self.shared_X = th.shared(np.asarray(self.X, dtype = th.config.floatX))
        self.shared_Y = th.shared(np.asarray(self.Y, dtype = th.config.floatX))



        
        
    
