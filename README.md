# randomly
Python3 Package for denoising single-cell data  with the  Random Matrix Theory. Please see example.ipynb for the denosing and visualization pipeline.
**Install Dependencies**

```
pip install -r requirements.txt
```

**Example of usage**

df - pandas dataframe : Pandas dataframe, shape (n_cells, n_genes)
index correspond the cell name and columns correspond the unique gene names

return denoised df2 pandas datarame for download and 
3 plots:  
a) Marchnko-Pastur distribuiton plot (mp.pdf)   
b) Gene statistics (statistics.pdf)  
c) t-SNE plot of denoised data (tsne.pdf)  

```
import randomly

# load example
import pandas as pd
df = pd.read_table('Data/data.tsv', index_col=0)

# run code
model=randomly.Rm()
model.preprocess(df)
model.fit()
```

Plotting

```
model.plot_mp(path = '../Figures/mp.pdf')
model.plot_statistics(path = '../Figures/statistics.pdf')
model.fit_tsne()
model.plot(type='tsne', path='../Figures/tsne.pdf', s=2)


```
**Denoising data**
To obtaining denoised data in the form of the pandas dataframe
(cells, signal genes), where the number of the signal genes is controlled with the false discovery rate fdr (fdr = 1) corresponds to all genes (default fdr = 0.001)

```
df2=model.return_cleaned(fdr=0.001)
