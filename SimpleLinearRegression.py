import numpy as np
import multiprocessing as mp

def func(x, m, b):
    return m * x + b


def ols_lls(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y)[0]
    return m, b


def ols_sums(x, y):
    n = len(x)
    sum_x = float(np.sum(x))
    sum_y = float(np.sum(y))
    sum_xx = float(np.sum(x*x))
    sum_xy = float(np.sum(x*y))
    m = (sum_xy - sum_x*sum_y / n) / (sum_xx - (np.power(sum_x, 2) / n))
    b = (sum_y / n) - m * (sum_x / n)
    return m, b

def ols_cov(x, y):
    m = np.cov(x, y)[0][1] / np.var(x, ddof=1)
    b = np.mean(y) - m * np.mean(x)
    return m, b

def map_func(f):
    age = int(f[17:19]) #18-19
    wtlbs = int(f[1950:1953]) #1951-1953 wt in lbs, self reported !888! !999!
    return [age, wtlbs]


if __name__ == '__main__':
    offset = 0
    pool = mp.Pool(processes=3)
    f = open('adult.dat')
    m = pool.map(map_func, f)
    for i, item in enumerate(list(zip(*m)[1])):
        if item == 888 or item == 999:
            m.pop(i - offset)
            offset += 1
    x = list(zip(*m)[0])
    y = list(zip(*m)[1])
    print pool.apply(ols_sums, (np.array(x), np.array(y)))
    print ols_cov(x, y)
    m, b = ols_lls(np.array(x), np.array(y))
    print m, b

    import matplotlib.pyplot as plt
    plt.plot(np.array(x), np.array(y), 'o', label='Original data', markersize=2)
    plt.plot(np.array(x), m*np.array(x) + b, 'r', label='Fitted line')
    plt.legend()
    plt.show()
