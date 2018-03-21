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

    



