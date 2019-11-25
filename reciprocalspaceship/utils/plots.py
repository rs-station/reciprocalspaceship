import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def plot_reciprocal_space_coverage(crystal, **kw):
    h,k,l = np.vstack(crystal.index.values).T
    hlim = np.abs(h).max()
    klim = np.abs(k).max()
    llim = np.abs(l).max()

    figsize = kw.get('figsize', (10, 10))
    f, ax = plt.subplots(2, 2, figsize=figsize)

    plt.sca(ax[0,0])
    plt.plot(h, k, 'ko', alpha=0.1)
    plt.xlim(-hlim, hlim)
    plt.ylim(-klim, klim)
    plt.ylabel("K")

    plt.sca(ax[1,0])
    plt.plot(h, l, 'ko', alpha=0.1)
    plt.xlim(-hlim, hlim)
    plt.ylim(-llim, llim)
    plt.xlabel("H")
    plt.ylabel("L")

    plt.sca(ax[0,1])
    plt.plot(l, k, 'ko', alpha=0.1)
    plt.xlim(-llim, llim)
    plt.ylim(-klim, klim)
    plt.xlabel("L")
    plt.show()
