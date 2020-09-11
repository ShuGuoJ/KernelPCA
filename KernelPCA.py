import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import  pdist,squareform,cdist
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class KPCA(object):
    def __init__(self,  n_dims, kernel='linear', gamma=1):
        super(KPCA,self).__init__()
        self.kernel = kernel
        self.n_dims = n_dims
        self.gamma = gamma

    def fit(self, data):
        # data:[features,n_samples]
        n = data.shape[1]
        self.data = data
        if self.kernel == 'linear':
            K = data.T@data
        elif self.kernel== 'rbf':
            K = self.rbf_kernel(data, self.gamma)
        else:
            K = None
        one_n = np.ones((n,n), dtype=np.float) / n
        # 聚集
        K_hat = K - one_n@K - K@one_n + one_n@K@one_n
        U,S,V = np.linalg.svd(K_hat)
        # W:[n_samples. n_dims]
        self.W = U[:,:self.n_dims]/S[:self.n_dims]

    def rbf_kernel(self,x,gamma):
        # data:[features,n_samples]=>[n_samples,features]=>[n_samples,1,features]
        # x:[features,n_samples]=>[n_samples,features]=>[1,n_samples,features]
        # dist = squareform(pdist(self.train_X))
        # K = self.rbf_kernel(dist)
        data = np.expand_dims(self.data.T, axis=1)
        x = np.expand_dims(x.T, axis=0)
        dist = data - x
        dist = np.sqrt(np.sum(np.power(dist,2),axis=-1))
        return np.exp(-dist/(2*gamma**2))

    def transform(self, samples):
        # samples:[features,n_samples]
        if self.kernel == 'linear':
            K = self.data.T @ samples
        elif self.kernel == 'rbf':
            K = self.rbf_kernel(samples, 15)
        m, n = samples.shape[1], self.data.shape[1]
        # one_mn:[m,n]
        one_mn = np.ones((m,n),dtype=np.float) / n
        # 聚集
        K_hat = K - one_mn @ K - K @ one_mn + one_mn @ K @ one_mn
        return self.W.T@K_hat