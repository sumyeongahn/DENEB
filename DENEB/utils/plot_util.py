import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



def scatter_hist(x,y,legend, save_name):
    for idx in range(len(x)):
        x[idx] = np.array(x[idx])
        y[idx] = np.array(y[idx])
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # the scatter plot:
    for idx in range(len(x)):
        ax.scatter(x[idx], y[idx], alpha=0.5, label = legend[idx])
    
    ax.legend()
    # Set aspect of the main axes.
    ax.set_aspect(1.)

    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(ax)
    # below height and pad are in inches
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    # now determine nice limits by hand:
    binwidth = 0.2
    
    xymax = 3.


    ax.set_xlim([0, xymax])
    ax.set_ylim([0, xymax])

    lim = (int(xymax/binwidth) + 1)*binwidth

    bins = np.arange(0, lim + binwidth, binwidth)
    for idx in range(len(x)):
        ax_histx.hist(x[idx], bins=bins, alpha=0.5, density=True)
        ax_histy.hist(y[idx], bins=bins, alpha=0.5, density=True, orientation='horizontal')


        

    plt.savefig(save_name)
    plt.close()


if __name__ == '__main__':
    # the random data
    x = [np.random.randn(1000),np.random.randn(1000),np.random.randn(1000)]
    y = [np.random.randn(1000),np.random.randn(1000),np.random.randn(1000)]
    scatter_hist(x,y, 'test.png')
    