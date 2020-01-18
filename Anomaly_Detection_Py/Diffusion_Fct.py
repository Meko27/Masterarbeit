'''
Module to calculate local distance matrix and weighted graph Laplacian

'''
import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
from numpy.matlib import repmat
import timeit
def local_dist_mat(x,k,metric='euclidean',grounddist=0,iteration=0):
# local_dist_mat: Calculates the knn - local distance matrix based on parameter k

    n_samples,n_dim = x.shape
    ixx = np.linspace(0,n_samples-1,n_samples,dtype=int)
    ixx = repmat(ixx,k,1)
    
    # check wich metric to be calculated
    if metric == 'emd':
        x = np.float32(x)
        grounddist = np.float32(grounddist)
        n_pdist = int(n_samples*(n_samples-1)/2) 
        dist_vect = np.empty((n_pdist))
        idx = 0
        for j in range(n_samples-1):
            start = j+1
            for i in range(start,n_samples):
                print('Now calculating {} of {} at iteration {}'.format(idx+1,n_pdist,iteration+1))
                start = timeit.timeit()
                dist_vect[idx],_,_ = cv2.EMD(np.transpose(x[i,:]),np.transpose(x[j,:]),cv2.DIST_USER,cost=grounddist)
                end = timeit.timeit()
                print('time elaplsed: ', end-start)
                idx = idx+1
    else:
        dist_vect = pdist(x,metric)
    
    dist_mat = squareform(dist_vect) 
    
    # calculate local dist matrix 
    knn_mat = np.argsort(dist_mat,axis=0) 
    knn_mat = knn_mat[1:k+1,:] # only first k elements (without the zeros)
    knn_dist_mat = np.zeros((k,n_samples))
    for j in range(n_samples):
        for i in range(k):
            idx_row = knn_mat[i,j]
            idx_col = j
            knn_dist_mat[i,j] = dist_mat[idx_row,idx_col]

    dist_mat_local = np.zeros((n_samples,n_samples))
    for j in range(n_samples):
        for i in range(k):
            idx_row = knn_mat[i,j]
            idx_col = ixx[i,j]
            dist_mat_local[idx_row,idx_col] = knn_dist_mat[i,j]

    dist_mat_local = np.maximum(dist_mat_local,np.transpose(dist_mat_local)) # symmetrize

    return dist_mat_local



def weighted_graph_laplacian(dist_mat):
# local_graph_laplacian: Calculates the weighted graph laplacian of a distance matrix
    n = dist_mat.shape[0]
    d = np.sum(dist_mat,axis=1) # sum up rows (-->)
    if len(d[d==0]) > 0:
        i_zeros = np.where(d==0)
        d[i_zeros] = 1
    D = np.diag(1/d) # Degree Matrix
    A = 1/n * np.matmul(np.matmul(D,dist_mat),D) # weighted Adjacency matrix
    d = np.sum(A,axis=1) # sum up rows (-->)
    if len(d[d==0]) > 0:
        i_zeros = np.where(d==0)
        d[i_zeros] = 1/n
    D = np.diag(d) # Degree Matrix of weighted A

    L = D-A # Graph Laplacian

    return L