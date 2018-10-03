# -*- coding: utf-8 -*-

"""Implementations of several clustering Algorithms based on scikit-learn library
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from collections import defaultdict

class Cluster():
    
    def __init__(self):
        self.X=None
    def fit_kmeans(self, n_clusters=2, random_state=1):
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(self.X)
        self.labels=kmeans_model.labels_
        
