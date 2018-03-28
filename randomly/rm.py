import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
from sklearn import preprocessing
from scipy import linalg as LA
from scipy import stats 
from scipy.stats import chi2
from scipy import linalg
import scipy.special
from scipy.linalg import qr 
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn import cluster, preprocessing, manifold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

style.use("ggplot")
np.seterr(invalid='ignore')
sns.set_style("white")
sns.set_context("paper")
sns.set_palette("deep")


class rm():
    """Random Matrix Theory Analysis of Principal components

    Parameters
    ----------
    svd_solver : string {'full', 'arpack', 'randomized'}
          full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < X.shape[1]
        randomized :
            run randomized SVD by the method of Halko et al.
    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.
    
    preprocess: True if the data needs preprocessing
    
    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.
    
    Attributes
    ----------
    components_ : array, shape (n_components, n_genes)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.
    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.
        Equal to n_components largest eigenvalues
        of the covariance matrix of X.
     
    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0.
       
    mean_ : array, shape (n_features,)
        Per-gene empirical mean, estimated from the original set.
    
    noise_variance_ : float
        The estimated noise covariance
    cell_names : list of cell names
    gene_names: list of gene nemes
    n_cells: number of cells 
    n_genes: number of genes"""

    def __init__(self, solver='wishart', tol=0.0, random_state=None):
            self.solver = solver
            self.tol = tol
            self.random_state = random_state
            
    def fit(self, df):
        """Fit RM model
        Parameters
        ----------
        df : Pandas dataframe, shape (n_cells, n_genes)
            where n_cells in the number of cells
            and n_genes is the number of genes.
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(df)
        return self
        
    def _fit(self, df):
        """Fit the model for the dataframe df and apply the dimensionality reduction on df using Marchenko - Pastur
        filtering"""
        #self.sparsity_cells=_get_sparsity_cells(df)
        #self.sparsity_genes=_get_sparsity_genes(df)
        self.X=self._preprocess(df)
        n_cells, n_genes = self.X.shape
        self.n_cells=n_cells
        self.n_genes=n_genes

        """Dispatch to the right submethod depending on the chosen solver."""
        if self.solver=='wishart':
            Y=self._wishart_matrix(self.X)
            (self.L,self.V)=self._get_eigen(Y)
            Xr=self._random_matrix(self.X)
            Yr=self._wishart_matrix(Xr)
            (self.Lr,self.Vr)=self._get_eigen(Yr)
        
            self.explained_variance_ = (self.L ** 2) / (self.n_cells)
            self.total_variance_ = self.explained_variance_.sum()
        
            """Compute Marchenko - Marchenko eigenvalues"""
            self.L_mp=self._mp_calculation(self.L, self.Lr)
        
            """Compute Tracy - Widom critical eigenvalue"""
            self.lambda_c=self._tw()
            """Number of structure componens correspond to number of eigenvalues larger than lambda_c estimated by Tracy - Widom"""

            self.n_components=len(self.L[self.L>self.lambda_c])
        else: 
            print 'Solver is undefined, please use Wishart Matrix as solver'

    def _tw(self):
        gamma=self._mp_parameters(self.L_mp)['gamma']
        p=len(self.L)/gamma
        sigma=1.0/np.power(p,2.0/3.0)*np.power(gamma,5.0/6.0)*np.power((1+np.sqrt(gamma)),4.0/3.0)
        lambda_c=np.mean(self.L_mp)*(1+np.sqrt(gamma))**2+3*sigma
        self.gamma=gamma
        self.p=p
        self.sigma=sigma
        return lambda_c


    def _wishart_matrix(self,X):
        """Compute Wishart Matrix of the Cells"""
        return np.dot(X,X.T)/(X.shape[1]+0.0)
    
    def _get_eigen(self,Y):
         """Compute Eigenvalues of the real symmetric matrix"""
        (L,V) = LA.eigh(Y)
        return (L,V)

    def _project_(self, df, V):
        
        return np.dot(df.values, V.T)

    def _random_matrix(self,X):
        return np.apply_along_axis(np.random.permutation, 0, X)
        
    def _to_tpm(self,df):
        df2=df.T/(df.T.sum()+0.0)*10**(6)
        return df2.T
        
    def _preprocess(self,df):
        """The method executes preprocessing of the data by removing the genes and cells that have 
        less than 5 transcripts. Transcripts are being converted to tpm and log2 scale is being taken.
        Genes are standard-normalized to have zero mean and standard deviation equal to 1.
        """
        self.gene_names=df.columns.tolist()
        self.cell_names=df.index.tolist()
        self.signal_genes=df.columns[df.sum()>10].tolist()
        self.signal_cells=df.index[df.T.sum()>10].tolist()
        self.filtered_genes=df.columns[df.sum()<=10].tolist()
        self.filtered_cells=df.index[df.T.sum()<=10].tolist()
        df=df.loc[self.signal_cells,self.signal_genes]
        df=self._to_tpm(df)
        df=np.log2(1+df)
        self.mean_=df.mean().values
        self.std_=df.std(ddof=0).values
        X=preprocessing.scale(df.values)
        return X



    def _mp_parameters(self, L):
        """Compute Parameters of the Marchenko Pastur Distribution of eigenvalues L"""
        moment_1=np.mean(L)
        moment_2=np.mean(np.power(L, 2))
        gamma=moment_2/float(moment_1**2)-1
        s=moment_1
        sigma=moment_2
        b_plus=s*(1+np.sqrt(gamma))**2
        b_minus=s*(1-np.sqrt(gamma))**2
        x_peak=s*(1.0-gamma)**2.0/(1.0+gamma)
        dic={'moment_1':moment_1,'moment_2':moment_2,'gamma':gamma,'b_plus':b_plus,'b_minus':b_minus,'s':s,'peak': x_peak}
        return dic

    def _marchenko_pastur(self,x,dic):
        #For distribution of eigenvalues
        pdf=np.sqrt((dic['b_plus']-x)*(x-dic['b_minus']))/float(2*dic['s']*np.pi*dic['gamma']*x)
        return pdf

    def _mp_pdf(self,x,L):
        vfunc=np.vectorize(self._marchenko_pastur)
        y=vfunc(x,self._mp_parameters(L))
        return y

    def _mp_calculation(self, L, Lr, eta=1, eps=10**-6, max_iter=1000):
        converged = False
        iter = 0
        loss_history = []
        b_plus=self._mp_parameters(Lr)['b_plus']
        b_minus=self._mp_parameters(Lr)['b_minus']
        L_updated=L[(L>b_minus) & (L<b_plus)]
        new_b_plus=self._mp_parameters(L_updated)['b_plus']
        new_b_minus=self._mp_parameters(L_updated)['b_minus']
        while not converged:
            loss=(1-float(new_b_plus)/float(b_plus))**2
            loss_history.append(loss)
            iter += 1 
            if loss <= eps:
                #print 'Converged, iterations:',iter
                converged = True
            elif iter == max_iter:
                print 'Max interactions exceeded!'
                converged = True
            else:
                gradient=new_b_plus-b_plus
                new_b_plus=b_plus+eta*gradient
                L_updated=L[(L>new_b_minus) & (L<new_b_plus)]
                b_plus=new_b_plus
                b_minus=new_b_minus
                new_b_plus=self._mp_parameters(L_updated)['b_plus']    
                new_b_minus=self._mp_parameters(L_updated)['b_minus']
        self.b_plus=new_b_plus
        self.b_minus=new_b_minus
        return L[(L>new_b_minus) & (L<new_b_plus)]
      

    def plot_mp(self,comparison=True, path=False, info=False):
        """Plot Eigenvalues,  Marchenko - Pastur distribution, randomized data and estimated Marchenko - Pastur for 
        randomized datas
        """
        x=np.linspace(0,int(round(np.max(self.L_mp)+0.5)),1000)
        y=self._mp_pdf(x,self.L_mp)
        yr=self._mp_pdf(x,self.Lr)
        
        if info:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            plt.figure()
        plot=sns.distplot(self.L, bins=800,norm_hist=True,kde=False,hist_kws={"alpha": 0.85,\
                                                                           "color":sns.xkcd_rgb["cornflower blue"]})   
        
        plot.set(xlabel='First cell eigenvalues normalized distribution')
        plt.plot(x,y,sns.xkcd_rgb["pale red"],lw=2)
        
        if comparison:  
            sns.distplot(self.Lr,bins=30,norm_hist=True, kde=False,hist_kws={"histtype": "step","linewidth": 3,\
                                                                      "alpha": 0.75,"color":sns.xkcd_rgb["apple green"]})
            plt.plot(x,yr, sns.xkcd_rgb["sap green"],lw=1.5,ls='--')
            plt.legend(['MP for random part in data','MP for randomized data','Randomized data','Real data']\
                       , loc="upper right",frameon=True)    
        else: 
            plt.legend(['MP for random part in data','Real data'], loc="upper right",frameon=True)        
        
        plt.xlim([0,int(round(np.max(self.L_mp)+0.5))])
        plt.grid()
        
        if info:
            dic=self._mp_parameters(self.L_mp)
            info1 = r'$\bf{Data\ Parameters}$'+'\n%i cells\n%i genes'\
                                    %(self.n_cells, self.n_genes)
            info2 = '\n'+r'$\bf{MP\ distribution\ in\ data}$'+'\n$\gamma=%.2f$\n$\sigma^2=%.2f$\n$b_-=%.2f$\n$b_+=%.2f$'\
                                    %(dic['gamma'],dic['s'],dic['b_minus'], dic['b_plus'])
            info3='\n'+r'$\bf{Analysis}$'+'\n%i eigenvalues > $\lambda_c (3 \sigma)$\n%i noise eigenvalues'\
                                    %(self.n_components, self.n_cells - self.n_components)
            infoT= info1+info2+info3
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            at=AnchoredText(infoT,loc=2, prop=dict(size=10), frameon=True, bbox_to_anchor=(1., 1.024),\
                           bbox_transform=ax.transAxes)
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            lgd=ax.add_artist(at)
            if path:    
                plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            if path:    
                plt.savefig(path)
        return plt.show()

    def plot(self, lib='All', type='mds', distance='',lib_name='Type', path=False, perplexity=30, n_comp=False): 
        #Visualizes data using 2d MDS, PCA, TSNE
        X_std=self.X#.StandardScaler().fit_transform(X)#First we normalize the data by Zscore
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
            plt.scatter(pos[:, 0], pos[:, 1], s=1, c='k')
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
            plt.scatter(pos[:, 0], pos[:, 1], s=1, c='k')
            plt.setp(ax.spines.values(), linewidth=0)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.xlabel('PC1, {0}%'.format(np.round(sklearn_pca.explained_variance_ratio_[0]*100)))
            plt.ylabel('PC2, {0}%'.format(np.round(sklearn_pca.explained_variance_ratio_[1]*100)))
            
        elif type=='tsne':
            if n_comp:
                pca = PCA(n_components=n_comp)
                X_std = pca.fit_transform(X_std)
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,metric='correlation', perplexity=perplexity)
            pos = tsne.fit_transform(X_std)
            
            fig = plt.figure(dpi=100, figsize=(5,5))
            ax = plt.gca()
            plt.scatter(pos[:, 0], pos[:, 1], s=1, c='k')
            plt.setp(ax.spines.values(), linewidth=0)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.xlabel('t-SNE1')
            plt.ylabel('t-SNE2')
        

        else: 
            print "Other Dimensionality reduction techniques are not implemented"
        if path:
            plt.savefig(path)
        
        return plt.show()

        



