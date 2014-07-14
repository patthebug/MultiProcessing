### Simple Linear Regression using Python Multiprocessing library

I recently learned executing "Simple Linear Regression" using `multiprocessing` module in python. This module is used for parallel processing in python. The `β` for Simple Linear Regression is calculated using the following formula:

$\beta  = \frac {\sum_{i=1}^{n} x_{i}y_{i} -  \frac {1}{n} \sum_{i=1}^{n} x_{i} \sum_{j=1}^{n} y_{j}} {\sum_{i=1}^{n} x_{i}^{2} - \frac {1}{n}(\sum_{i=1}^{n} x_{i})^2}$

I will use the `multiprocessing` and `numpy` modules for this task.

```
import numpy as np
import multiprocessing as mp
```

The dataset that I will be using for this task is freely downloadable from [here](ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/nhanes/nhanes3/1A/adult.dat).  

I will regress `age` against `weight`. The following function may be used to read the file in:

```
def map_func(f):
    age = int(f[17:19]) #18-19
    wtlbs = int(f[1950:1953]) #1951-1953 wt in lbs, self reported !888! !999!
    return [age, wtlbs]
```

I am going to use this function to read the file in parallel using 3 processes at the same time. I am using a quad-core machine which is why I have left 1 process untouched, for recovery purposes just in case I happen to write an infinite loop.

For achieving this, I will use the `Pool` function and initiate 3 processes. After creating these processes, I will use the `map` function to distribute the reading job amongst the 3 processes. 

```
pool = mp.Pool(processes=3)
f = open('adult.dat')
m = pool.map(map_func, f)
```

The data has some observations in which the weight was unknown or not disclosed. Such observations have weight values of 888 and 999 and the following code will get rid of these observations.

```
for i, item in enumerate(list(zip(*m)[1])):
        if item == 888 or item == 999:
            m.pop(i - offset)
            offset += 1

x = list(zip(*m)[0])
    y = list(zip(*m)[1])
```

The above code now leaves us with `age` and `weight` vectors in variables `x` and `y` respectively. Let us first up calculate the coefficients for simple linear regression using the conventional method (finding the least squares fit). 

```
def ols_lls(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y)[0]
    return m, b

m, b = ols_lls(np.array(x), np.array(y))
print m, b
# should print the values: (-0.0171020631488, 163.871290024)
```

It may be a good idea now to verify these values with our own implementation of simple linear regression using the formula presented earlier.

```
def ols_sums(x, y):
    n = len(x)
    sum_x = float(np.sum(x))
    sum_y = float(np.sum(y))
    sum_xx = float(np.sum(x*x))
    sum_xy = float(np.sum(x*y))
    m = (sum_xy - sum_x*sum_y / n) / (sum_xx - (np.power(sum_x, 2) / n))
    b = (sum_y / n) - m * (sum_x / n)
    return m, b

print pool.apply(ols_sums, (np.array(x), np.array(y)))
# should print the values: (-0.017102063148779986, 163.87129002440847)
```

Notice that I have used two different functions `map` and `apply` available within the `multiprocessing` module. More about these methods can be learnt [here](https://docs.python.org/dev/library/multiprocessing.html).

Finally, one can easily notice that the values of parameters using the conventional `numpy` method and `multiprocessing` module are almost exactly the same.

Now for finally plotting the results, I will use `matplotlib`.

```
import matplotlib.pyplot as plt
plt.plot(np.array(x), np.array(y), 'o', label='Original data', markersize=2)
plt.plot(np.array(x), m*np.array(x) + b, 'r', label='Fitted line')
plt.legend()
plt.show()
```

