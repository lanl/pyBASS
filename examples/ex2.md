In this example, we generate data from the Friedman function but treat one of the variables (the first) as a functional variable.


```python
import pyBASS as pb
import numpy as np
import matplotlib.pyplot as plt
```

This generates 500 curves of length 50.


```python
# Friedman function where first variable is the functional variable
def f2(x):
    out = 10. * np.sin(np.pi * tt * x[1]) + 20. * (x[2] - .5) ** 2 + 10 * x[3] + 5. * x[4]
    return out

tt = np.linspace(0, 1, 50) # functional variable grid
n = 500 # sample size
p = 9 # number of predictors other (only 4 are used)
x = np.random.rand(n, p) # training inputs
xx = np.random.rand(1000, p)
e = np.random.normal(size=[n, len(tt)]) * .1 # noise
y = np.apply_along_axis(f2, 1, x) + e # training response
ftest = np.apply_along_axis(f2, 1, xx)
ytest = ftest + np.random.normal(size=[1000, len(tt)]) * .1

plt.plot(y.T)
plt.show()
```


    
![png](ex2_files/ex2_3_0.png)
    


Now take the principal component decomposition of y to see that most of the variation is captured with 3 principal components.


```python
pca = pb.BassPCAsetup(y)
pca.plot()
```


    
![png](ex2_files/ex2_5_0.png)
    


Now fit a BASS model in PCA space.


```python
mod2 = pb.bassPCA(x, y, npc=3, ncores=3)
mod2.plot()
```

    Starting bassPCA with 3 components, using 3 cores.
    BASS MCMC Complete. Time: 3.399235 seconds.
    BASS MCMC Complete. Time: 3.402881 seconds.
    BASS MCMC Complete. Time: 3.504653 seconds.



    
![png](ex2_files/ex2_7_1.png)
    


Here are posterior mean predictions of our test data:


```python
fpred = mod2.predict(xx)
plt.plot(fpred.mean(0).T)
plt.show()
```


    
![png](ex2_files/ex2_9_0.png)
    


Test data posterior mean vs noiseless observed:


```python
plt.plot(fpred.mean(0).T, ftest.T)
plt.show()
```


    
![png](ex2_files/ex2_11_0.png)
    


Functional residuals:


```python
plt.plot(fpred.mean(0).T - ftest.T)
plt.show()
```


    
![png](ex2_files/ex2_13_0.png)
    


Prediction uncertainty for one of the test samples:


```python
ypred = mod2.predict(xx, nugget=True, trunc_error=False)
plt.plot(ypred[:,70,:].T, color='grey')
plt.plot(ytest[70])
plt.show()
```


    
![png](ex2_files/ex2_15_0.png)
    

