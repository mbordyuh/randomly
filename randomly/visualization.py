"""Visualization of Marchenko Pastur algorithm and implementation
MDS, PCA, tsne dimensionality reduction techniques for visualization
"""
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import host_subplot

from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib import style
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn import cluster, preprocessing, manifold
from sklearn.decomposition import PCA
import pandas as pd

style.use("ggplot")
np.seterr(invalid='ignore')
sns.set_style("white")
sns.set_context("paper")
sns.set_palette("deep")

class visualize():
    
    def __init__(self):
        pass
    
    def plot(self, X, labels=False, type='tsne', distance='',path=False, perplexity=30, n_comp=False, s=3, c='k'): 
        #Visualizes data using 2d MDS, PCA, TSNE
        if not isinstance(labels, bool):
            labels=list(labels)

        X_std=X#.StandardScaler().fit_transform(X)#First we normalize the data by Zscore
        if type=='mds':
            if distance=='precomputed':
                similarities=X
            else:
                similarities = 1-np.corrcoef(X_std)
       
            mds = manifold.MDS(n_components=2, max_iter=1000, eps=1e-8,
                           dissimilarity="precomputed", n_jobs=1)
            pos = mds.fit(similarities).embedding_
            clf = PCA(n_components=2)
            pos = clf.fit_transform(pos)
            fig = plt.figure(dpi=100, figsize=(5,5))
            ax = plt.gca()
            if labels:
                for j in np.unique(labels):
                    plt.scatter(pos[labels==j, 0], pos[labels==j, 1], s=s, c=c)
            else:    
                plt.scatter(pos[:, 0], pos[:, 1], s=s, c=c)
            plt.setp(ax.spines.values(), linewidth=0)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.xlabel('MDS1')
            plt.ylabel('MDS2')
        elif type=='pca':
            sklearn_pca = PCA(n_components=2)
            pos = sklearn_pca.fit_transform(X_std)
            
            fig = plt.figure(dpi=100, figsize=(5,5))
            ax = plt.gca()
            if labels:
                for j in np.unique(labels):
                    plt.scatter(pos[labels==j, 0], pos[labels==j, 1], s=s)
            else:    
                plt.scatter(pos[:, 0], pos[:, 1], s=s, c=c)
            plt.setp(ax.spines.values(), linewidth=0)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.xlabel('PC1, {0}%'.format(np.round(sklearn_pca.explained_variance_ratio_[0]*100)))
            plt.ylabel('PC2, {0}%'.format(np.round(sklearn_pca.explained_variance_ratio_[1]*100)))
            
        elif type=='tsne':
            if n_comp:
                pca = PCA(n_components=n_comp)
                X_std = pca.fit_transform(X_std)
            tsne = manifold.TSNE(n_components=2, init='pca', 
                                 random_state=0,metric='correlation', 
                                 perplexity=perplexity)
            pos = tsne.fit_transform(X_std)
            
            fig = plt.figure(dpi=100, figsize=(5,5))
            ax = plt.gca()
            if labels:
                for j in np.unique(labels):
                    plt.scatter(pos[labels==j, 0], pos[labels==j, 1], s=s)
            else:    
                plt.scatter(pos[:, 0], pos[:, 1], s=s, c=c)
            
            plt.setp(ax.spines.values(), linewidth=0)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.xlabel('t-SNE1')
            plt.ylabel('t-SNE2')
            #plt.axis('off')
        if path:
            plt.savefig(path)
        return plt.show()





