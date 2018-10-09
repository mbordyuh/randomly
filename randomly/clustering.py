"""Implementations of several clustering
algorithms based on scikit-learn library
"""

from sklearn.cluster import KMeans


class Cluster():
    '''Attributes
       ----------

       X: array-like or sparse matrix, shape=(n_cells, n_genes)
          Training instances to clustering

       labels:
       Labels for each data point
    '''
    
    def __init__(self):
        self.X = None

    def fit_kmeans(self, n_clusters=2, random_state=1):
        kmeans_model = KMeans(n_clusters=n_clusters,
                              random_state=1).fit(self.X)
        self.labels = kmeans_model.labels_
