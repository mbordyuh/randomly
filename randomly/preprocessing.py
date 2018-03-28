import pandas as pd
import numpy as np

def _to_tpm(df):
        df2=df.T/(df.sum(axis=1)+0.0)*10**(6)
        return df2.T
    
def preprocess(df,preprocessing, min_tpm=1.0):
    """The method executes preprocessing of the data by removing the genes 
    and cells that have less than 10 transcripts by default. Transcripts are 
    being converted to tpm and log2 scale is being taken. Genes are 
    standard-normalized to have zero mean and standard deviation equal to 1.
    Input pandas DataFrame: Pandas dataframe, shape (n_cells, n_genes)
    Returned preprocessed Dataframe
    """
    if preprocessing=='sc':
        signal_genes=df.columns[df.sum()>min_tpm].tolist()
        signal_cells=df.index[df.T.sum()>min_tpm].tolist()
        filtered_genes=df.columns[df.sum()<=min_tpm].tolist()
        filtered_cells=df.index[df.T.sum()<=min_tpm].tolist()
        df=df.loc[signal_cells, signal_genes]
        df=_to_tpm(df)
        df=np.log2(1+df)
    df=(df-df.mean())/(df.std(ddof=0)+0.0)
    return df



    