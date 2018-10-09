
"""
Visualization of Marchenko Pastur algorithm and implementation
PCA, tSNE dimensionality reduction techniques
"""
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.decomposition import PCA

import multiprocessing 
from MulticoreTSNE import MulticoreTSNE
import numpy as np
import pandas as pd
import seaborn as sns

from .palettes import pallete_50

style.use("ggplot")
np.seterr(invalid='ignore')
sns.set_style("white")
sns.set_context("paper")
sns.set_palette("deep")
sns.set_style( {'axes.linewidth': 0.5})


class Visualize():
    '''class for embedding visualization'''
    def __init__(self):
        self.X=None
        self.position_flag=False
    
    def fit_tsne(self,
                perplexity=30,
                learning_rate =1000,
                early_exaggeration=12,
                pca=True,
                n_comp=False,
                multicore=True,
                fdr=1): 
        """
        Embedding of single cell data with t-distributed stochastic neighborhood 
        embedding (tSNE) for 2D visualization. By default, we use Multicore-tsne 
        implementation by Dmitry Ulyanov 
        https://github.com/DmitryUlyanov/Multicore-TSNE>, 
        if number of processors is great than 1. Otherwise, scikit-learn 
        implementation is used. Default parametes by scikit-learn are used.

        Parameters
        ----------
        perplexity : float, optional (default: 30)

        The perplexity is related to the number of nearest neighbors that 
        is used in other manifold learning algorithms. Larger datasets usually 
        require a larger perplexity. Consider selecting a value between 5 and 50.
        The choice is not extremely critical since t-SNE is quite insensitive to 
        this parameter.

        early_exaggeration : 'float', optional (default: 12.0)
            Controls how tight natural clusters in the original space are in the
            embedded space and how much space will be between them. For larger
            values, the space between natural clusters will be larger in the
            embedded space. Again, the choice of this parameter is not very
            critical. If the cost function increases during initial optimization,
            the early exaggeration factor or the learning rate might be too high.

        learning_rate : 'float', optional (default: 1000)
            Note that the R-package "Rtsne" uses a default of 200.
            The learning rate can be a critical parameter. It should be
            between 100 and 1000. If the cost function increases during initial
            optimization, the early exaggeration factor or the learning rate
            might be too high. If the cost function gets stuck in a bad local
            minimum increasing the learning rate helps sometimes.

        Returns
        -------
        Updated insance self, with self.embedding containing 2D t-SNE coordianates
        """

        if self.X is None:
            raise ValueError('Nothing to plot, please fit the data first')
        else:
            genes=self.select_genes(fdr)
            X=self.X.copy()[:,self.normal_genes.isin(genes)]
                            

        if n_comp:
            pca = PCA(n_components=n_comp, svd_solver='full')
            X = pca.fit_transform(X)

        n_jobs = multiprocessing.cpu_count()

        if n_jobs > 1 and multicore:
            tsne = MulticoreTSNE(n_jobs=n_jobs, 
                                 perplexity=perplexity)

            print('computing t-SNE, using Multicore t-SNE for {0} jobs'.format(n_jobs))
            # need to transform to float64 for MulticoreTSNE...
            self.embedding = tsne.fit_transform(X.astype('float64'))
        else:
            print('computing t-SNE, using scikit-learn implementation')
            tsne = manifold.TSNE(n_components=2,
                                  init='pca',
                                  random_state=0,
                                  metric='correlation',
                                  perplexity=perplexity)

            self.embedding = tsne.fit_transform(X)
        print('atribute embedding is updated with t-SNE coordinates')
        return
    
    def fit_pca(self, n_comp=2, fdr=1): 
        
        '''2D PCA of the Data based on first 2 principal components'''
        if self.X is None:
            raise ValueError('Nothing to plot, please fit the data first')
        else:
            genes=self.select_genes(fdr)
            X=self.X.copy()[:, self.normal_genes.isin(genes)]
           
        pca = PCA(n_components=n_comp, svd_solver='full')
        self.embedding=pca.fit_transform(self.X)
        print('atribute embedding is updated with t-SNE coordinates')
        return
            
    def plot(self,
           path=False,
           title=False,
           labels=False,
           palette=pallete_50,
           gene=False,
           data=False,
           height=5,
           fontsize=10,
           legend=False,
           legendcol=5,
           psize=5,
           xytitle='t-SNE',
            ):
        "Ploting labels"
        
        if labels is not False:
            if not palette:
                palette=sns.color_palette("husl", len(set(labels))+1)
            
            
            with sns.plotting_context("paper", font_scale=1.5):
                
                fig=plt.figure(figsize=(height, height+0.5), dpi=100)
                
                sns.lmplot( x='x',
                            y='y',
                            fit_reg=False,
                            scatter_kws={'s': psize,
                                         'alpha': 1},
                            hue='label',
                            data=pd.DataFrame(self.embedding,
                                              columns=  ['x','y'])    
                                              .join(pd.Series(labels, name='label')),
                            height=height,
                            palette=sns.set_palette(palette),
                            legend=False)
                fig.set_tight_layout(False)
                
                if legend is not False:    
                         plt.legend(legend,
                                    loc='lower center',
                                    bbox_to_anchor=(0.5, 1.05),
                                    ncol=legendcol,
                                    frameon=True,
                                    markerscale=height//2,
                                    fontsize=height+3)     

                plt.xlabel(xytitle + '1', fontsize=fontsize)
                plt.ylabel(xytitle + '2', fontsize=fontsize)


        elif type(gene) is list:
             
            color=self.X[:, self.normal_genes.isin(gene)].mean(axis=1)
            with plt.style.context('seaborn-paper'):
                
                fig=plt.figure(figsize=(height, height + 0.5), dpi=100)
                fig.set_tight_layout(False)
            
                g=plt.scatter(self.embedding[:,0],
                              self.embedding[:,1],
                              s=psize,
                              c=color,
                              alpha=1,
                              cmap='coolwarm'
                              )
                plt.xlabel(xytitle + '1', fontsize=fontsize)
                plt.ylabel(xytitle + '2', fontsize=fontsize)
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
                plt.autoscale(enable=True, axis='both')
               
                
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes("right", "2.5%", pad="1%")
                
                if gene[0]=='library':
                    cb=plt.colorbar(g, cax=cax, label='library complexity', ticks=[])
                    cb.set_label(label='library complexity', fontsize=fontsize-2)
                else:
                    cb=plt.colorbar(g, cax=cax, label='log2(1+TPM)', ticks=[])
                    cb.set_label(label='log2(1+TPM)', fontsize=fontsize-2)
                
                
        elif type(gene) is tuple:
            n=len(gene)
            nrow = int(np.sqrt(n))
            ncol = int(np.ceil(n / nrow))
            
            if (n % 2 != 0 and n > 3) or nrow * ncol < n:
                ncol = ncol+1
            
            if n<4:
                fig, axs = plt.subplots(nrow, ncol, dpi=100,
                                        figsize=(ncol*height*1.5,
                                                 nrow*height*1.5)
                                        )
            else:
                fig, axs = plt.subplots(nrow, ncol, dpi=100,
                                        figsize=(ncol*height,
                                                 nrow*height)
                                        )

            if nrow*ncol>n:
                for i in range(ncol*nrow - n):
                    fig.delaxes(axs[-1][-(i+1)])      
            if type(axs) != np.ndarray:
                axs = [axs]
            else:
                axs = axs.ravel()
            for i in range(n):
                if i < n:
                    if type(gene[i]) is list:
                        marker = gene[i]
                    else:
                        marker = [gene[i]]
                    color = self.X[:, 
                                   self.normal_genes.isin([gene[i]])].mean(axis=1)        

                    with plt.style.context('seaborn-paper'):
                        g = axs[i].scatter(self.embedding[:, 0],
                                           self.embedding[:, 1],
                                           s=psize,
                                           c=color,
                                           alpha=1,
                                           cmap='coolwarm'
                                           )
                        axs[i].set_xticks([])
                        axs[i].set_yticks([])
                        axs[i].autoscale(enable=True, axis='both')

                        divider = make_axes_locatable(axs[i])
                        cax = divider.append_axes("right", "2.5%", pad="1%")

                        axs[i].set_title(str(marker[0]), fontsize=fontsize)
                        if marker[0] == 'library':
                            cb=fig.colorbar(g, cax=cax, 
                                            label='library complexity',
                                            ticks=[])
                            cb.set_label(label='library complexity',
                                         fontsize=fontsize-2)
                        else:
                            cb=fig.colorbar(g, cax=cax, 
                                            label='log2(1+TPM)',
                                            ticks=[])
                        cb.set_label(label='log2(1+TPM)',
                                     fontsize=fontsize-2)
                        
                        fig.set_tight_layout(False)
                    
                    if title:
                        axs[i].set_title(title)
                    else:
                        if len(marker) < 2:
                            axs[i].set_title(str(marker[0]),
                                             fontsize=fontsize)

                        elif len(marker) > 1:
                                axs[i].set_title('list starting with ' + str(marker[0]))

                    if i % ncol == 0:
                        axs[i].set_ylabel(xytitle+'2', fontsize=fontsize)
                    if ((i // ncol) + 1) == nrow:
                        axs[i].set_xlabel(xytitle+'1', fontsize=fontsize)

        else:
            with sns.plotting_context("paper", font_scale=1.5):
                sns.lmplot(x='x',
                           y='y',
                           fit_reg=False,
                           scatter_kws={'s': psize,
                                        'alpha': .9,                                                                                                                   'color':'black'},
                           hue=None,
                           data=pd.DataFrame(self.embedding,
                                             columns=['x', 'y']),
                           height=height,
                           aspect=1,
                           legend=False,
                           )
                plt.xlabel(xytitle + '1', fontsize=fontsize)
                plt.ylabel(xytitle + '2', fontsize=fontsize)

        sns.despine(top=False, right=False, left=False, bottom=False)

        if title:
            plt.title(title)
        if path:
            plt.savefig(path, bbox_inches='tight')
        plt.show()

        return
