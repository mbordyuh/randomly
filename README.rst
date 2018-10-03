========
Randomly
========

Python3 module for denoising single-cell data through Random Matrix Theory. Please see example.ipynb for the denoising and visualization pipeline.

.. image:: https://img.shields.io/pypi/v/randomly.svg
        :target: https://pypi.python.org/pypi/randomly

.. image:: https://img.shields.io/travis/luisaparicio/randomly.svg
        :target: https://travis-ci.org/luisaparicio/randomly

.. image:: https://readthedocs.org/projects/randomly/badge/?version=latest
        :target: https://randomly.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Authors
-------

Mykola Bordyuh, Luis Aparicio (equal contribution).

Installation
------------

.. code:: shell
    
    pip install --upgrade git+https://github.com/RabadanLab/randomly

Usage 
-----

Input parameters:

- df :
    pandas DataFrame, shape (n_cells, n_genes)

Results:

- df_denoised :
    pandas DataFrame, shape (n_cells, n_signal_genes)

Additional plots:  

- Marchenko-Pastur distribution plot
- Statistics on genes
- t-SNE plot of denoised data

**Preparation**

.. code:: python
    
    import pandas as pd
    import randomly

    # Data loading
    df = pd.read_table('Data/data.tsv', sep='\t', index_col=0)

    # Model fitting on input data
    model = randomly.Rm()
    model.preprocess(df)
    model.fit()


**Plotting**

.. code:: python

    model.plot_mp(path = 'Figures/mp.pdf')
    model.plot_statistics(path = 'Figures/statistics.pdf')
    model.fit_tsne()
    model.plot(path = 'Figures/tsne.pdf')


**Data Denoising**

Denoised data is returned through a pandas DataFrame of shape (cells, signal genes), where the number of signal genes is controlled through the False Discovery Rate parameter (fdr = 1 corresponds to all genes, default fdr = 0.001)

.. code:: python
    
    df_denoised = model.return_cleaned()


License and documentation
-------------------------

* Free software: MIT license
* Documentation (TODO): https://randomly.readthedocs.io.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
