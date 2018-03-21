import pandas as pd
import numpy as np
import os
import sys
from scipy import stats, linalg
from .preprocessing import preprocessing
from sklearn.decomposition import PCA


class rm():
    """Random Matrix Theory Analysis of Principal components
    Parameters
    ----------
    eigen_solver : string {'wishart'}. Find eigenvalue and eigenvectors 
        wishart : Compute wishart matrix and find eigenvalues and eigenvectors
        of the Wishart matrix
            
    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.
    
    preprocessing: string {'sc','False'}
            sc: run single cell preprocessing
            False: skip preprocessing
            
    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    
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
    
    gene_names: list of gene names
    
    X: 2D array 
        Preprocessed single cell matrix
    
    n_cells: int
    number of cells 
    
    n_genes: int
     number of genes 
    
    L: list
        Eigenvalues of the Wishart matrix
    V: 2D array
        Eigenvectors of the Wishart matrix

    Lr: list
        Eigenvalues of the Wishart randomized matrix
    
    Vr: 2D array
        Eigenvectors of the Wishart randomized matrix
    
    L_mp: list
        Estimated Marchenko - Pastur Eigenvalues

    lambda_c: float
         Critical eigenvalues estimated by Tracy - Widom distribution
    
    n_components: int
        Number of components above the Marchenko - Pastur distribution
    
    gamma: float 
            Estimated gamma of the Marchenko - Pastur distribuiton

    sigma: float
        Estimated sigma of the Marchenko - Pastur distribution
    
    b_plus: float 
        Estimated upper bound of the Marchenko - Pastur distribution

    b_minus: float
        Estimated lower bound of the Marchenko - Pastur distribution
    """


    def __init__(self, eigen_solver='wishart', tol=0.0, random_state=None, preprocessing='sc'):

        self.eigen_solver = eigen_solver
        self.preprocessing=preprocessing
        self.tol = tol
        self.random_state = random_state
        self.preprocessing
        self.n_cells=0
        self.n_genes=0
        self.cell_names=[]
        self.gene_names=[]
        self.L=[]
        self.V=None
        self.Lr=[]
        self.Vr=None
        self.explained_variance_=[]
        self.total_variance_=[]
        self.L_mp=[]
        self.lambda_c=0
        self.n_components=0
        self.gamma=0
        self.p=0
        self.sigma=0
        self.b_plus=0
        self.b_minus=0


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
        """Fit the model for the dataframe df and apply the dimensionality reduction on
        using Marchenko - Pastur filtering
        """
        
        if self.preprocessing=='sc':
            df=preprocessing(df)
        
        n_cells, n_genes = df.shape
        self.n_cells=n_cells
        self.n_genes=n_genes
        self.gene_names=df.columns.tolist()
        self.cell_names=df.index.tolist()
        self.X=df.values
        self.mean_=df.mean().values
        self.std_=df.std(ddof=0).values

        """Dispatch to the right submethod depending on the chosen solver"""
        if self.eigen_solver=='wishart':
            Y=self._wishart_matrix(self.X)
            (self.L,self.V)=self._get_eigen(Y)
            Xr=self._random_matrix(self.X)
            Yr=self._wishart_matrix(Xr)
            (self.Lr,self.Vr)=self._get_eigen(Yr)
        
            self.explained_variance_ = (self.L ** 2) / (self.n_cells)
            self.total_variance_ = self.explained_variance_.sum()
        
            self.L_mp=self._mp_calculation(self.L, self.Lr)
        
            self.lambda_c=self._tw()
           
            self.n_components=len(self.L[self.L>self.lambda_c])
        else: 
            print('Solver is undefined, please use Wishart Matrix as eigenvalue solver')
        pca = PCA(n_components=self.n_components)
        self.X_cleaned = pca.fit_transform(self.X)
    
    def return_cleaned(self):
        df=pd.DataFrame(self.X_cleaned)
        return df

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
        """Compute Wishart Matrix of the cells"""
        return np.dot(X,X.T)/(X.shape[1]+0.0)
    
    def _random_matrix(self,X):
        return np.apply_along_axis(np.random.permutation, 0, X)
        
    def _get_eigen(self,Y):
        """Compute Eigenvalues of the real symmetric matrix"""
        (L,V) = linalg.eigh(Y)
        return (L,V)

    def _project_(self, df, V):
        return np.dot(df.values, V.T)

   

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
        dic={'moment_1':moment_1,'moment_2':moment_2,'gamma':gamma,'b_plus':b_plus
            ,'b_minus':b_minus,'s':s,'peak': x_peak}
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
                print('Max interactions exceeded!')
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
      

    



