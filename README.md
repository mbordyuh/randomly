# randomly
Python Package for denoising single cell with Random Matrix 

**Install Dependencies**

```
pip install -r requirements.txt
```

**Example of usage**

df - pandas dataframe : Pandas dataframe, shape (n_cells, n_genes)
index correspond the cell name and columns correspond the unique gene names

return denoised df2 pandas datarame for download

```
import randomly

# load example
import pandas as pd
df = pd.read_table('Data/data.tsv')

# run code
model=randomly.rm()
model.preprocess(df)
model.fit()
```

Plotting

```
model.plot_mp(path='../Figures/mp_{0}.pdf'.format(name))
model.plot_statistics(path='../Figures/statistics_{0}.pdf'.format(name))
model.plot(model.X_cleaned, type='tsne', path='../Figures/tsne.cleaned_{0}.pdf'.format(name),s=2)
df2=model.return_cleaned()
```
