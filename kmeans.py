import numpy as np
import random

def distance(x1, x2):
    "euclidian distance"
    return np.linalg.norm(x1-x2)

def cal_centroid(arr):
    return np.sum(arr,axis=0)/len(arr)

def init_centroids(X,k):
    c = []
    c.append(random.randint(0,len(X)-1)) # first centroid index
    for i in range(k-1): # find the rest k-1 centers
        d = [0]*len(X)
        for index, x in enumerate(X):
            if index not in c: # except for those already are centroids
                # its min distance with all the known centroids
                d[index]= np.min([np.linalg.norm(x-X[s]) for s in c])
        c.append(np.argmax(d)) # the point that has largest distance_c_min
    return X[c]

def kmeans(X: np.ndarray, k:int, centroids=None, tolerance=1e-2):
    if centroids is None:
        # random initial centroids , t=0
        centroids = X[random.sample(range(len(X)),k)]
    elif centroids=='kmeans++':
        # make the initial centroids sparse
        centroids = init_centroids(X,k)
    
    new_centroids = np.ones(centroids.shape) # init it
    
    while True:
        clusters = [[] for i in range(k)]
        # list of list, each contains the obs index
        for index, x in enumerate(X):
            j = np.argmin([np.linalg.norm(x-center) for center in centroids])
            clusters[j].append(index) # record its index, not value
        for j in range(k):
            new_centroids[j] = cal_centroid(X[clusters[j]])
        if distance(centroids, new_centroids) < tolerance:
            break
        centroids = new_centroids

    return new_centroids, clusters

def leaf_samples(rf, X:np.ndarray):
    """
        Return a list of arrays where each array is the set of X sample indexes
        residing in a single leaf of some tree in rf forest. For example, if there
        are 4 leaves (in one or multiple trees), we might return:
        
        array([array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        array([10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19, 20]),
        array([21, 22, 23, 24, 25, 26, 27, 28, 29]))
        """
    n_trees = len(rf.estimators_)
    leaf_samples = []
    leaf_ids = rf.apply(X)  # which leaf does each X_i go to for sole tree?
    for t in range(n_trees):
        # Group by id and return sample indexes
        uniq_ids = np.unique(leaf_ids[:,t])
        sample_idxs_in_leaves = [np.where(leaf_ids[:, t] == id)[0] for id in uniq_ids]
        leaf_samples.extend(sample_idxs_in_leaves)
    return leaf_samples

def similarity_matrix(X, rf):
    """
        return how many times X[i] and X[j] appears in the same leaf
        """
    leaves = leaf_samples(rf,X)
    n = X.shape[0]
    S = np.zeros((n,n))
    observation_leaves = [[] for i in range(n)]
    # loop over the leaves
    for leaf_number,leaf in enumerate(leaves):
        for item in leaf:
            observation_leaves[item].append(leaf_number)
    # update S[i][j]
    for i in range(n):
        for j in range(i,n):
            S[i][j] = len(set(observation_leaves[i]).intersection(set(observation_leaves[j])))
            S[j][i] = S[i][j]
    return S
