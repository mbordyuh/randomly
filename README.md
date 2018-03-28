# randomly
Python Package for denoising single cell with Random Matrix 

**Example of usage**

df - pandas dataframe 

return denoised df2 pandas datarame for download

```
import randomly
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
