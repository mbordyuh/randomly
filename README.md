# randomly
Python Package for denoising single cell with Random Matrix 

**Example of usage**

df - pandas dataframe 

return denoised df2 pandas datarame for download

```
import randomly
model=randomly.rm()
model.fit(df)
df2=model.return_cleaned()
```
