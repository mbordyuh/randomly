# randomly

Python3 module for denoising single-cell data through Random Matrix Theory. Please see example.ipynb for the denoising and visualization pipeline.

**Installation Dependencies**

```shell
pip install -r requirements.txt
```

**Usage Example**

It is important to use Python3 (Python2 would not work)

Input parameters:
- df : pandas DataFrame, shape (n_cells, n_genes) where cell barcodes are stored in the index and gene symbols in the columns. Values in the table should be transcripts (integers).

Results:
- df_denoised : pandas DataFrame, shape (n_cells, n_signal_genes)

Additional plots:  
    
    a) Marchenko-Pastur distribution plot
    
    b) Statistics on genes
    
    c) t-SNE plot of denoised data

***Preparation***

```python
import pandas as pd
import randomly

# Data loading
df = pd.read_table('Data/data.tsv', sep='\t', index_col=0)

# Model fitting on input data
model = randomly.Rm()
model.preprocess(df, min_tp=10,
                 min_genes_per_cell=10,
                 min_cells_per_gene=10)
model.fit()
```
         """The method executes preprocessing of the data by removing 
        1. Genes and cells that have less than 10 transcripts by default. 
        2. Cells are removed that express less than min_tp genes
        3. Genes are removed that expressed less than in 10 cells

        4. Transcripts are being converted to log2(1+TPM). 
        5. Genes are  standard-normalized to have zero mean and standard 
        deviation equal to 1. 

***Plotting***

```python
model.plot_mp(path='Figures/mp.pdf')
model.plot_statistics(path='Figures/statistics.pdf')
model.fit_tsne()
model.plot(path='Figures/tsne.pdf')
```

***Data Denoising***

Denoised data is returned as a pandas DataFrame of shape (cells, signal genes), where the number of signal genes is controlled through the False Discovery Rate parameter (fdr=1 corresponds to all genes, default fdr= 0.001)

```python
df_denoised = model.return_cleaned()
```
