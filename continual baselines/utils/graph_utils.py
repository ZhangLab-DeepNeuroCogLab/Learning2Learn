import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import statistics
import pickle
import itertools
from scipy.stats import sem


def get_stderr(data, type='col'):
    data_mean = []
    data_stderr = []
    if type == 'col':
        cols = len(data[0])
        rows = len(data)
        for col in range(cols):
            temp_data = []
            for row in range(rows):
                temp_data.append(data[row][col])
            data_mean.append(statistics.mean(temp_data))
            data_stderr.append(sem(temp_data))

    return data_mean, data_stderr


def plot_errorbar(x, y, x_label, y_label, x_ticks, y_ticks, title=None, error=None, save_loc=None):
    if error is None:
        error = [0] * len(y)

    fig, ax0 = plt.subplots(nrows=1, sharex=True, figsize=(5, 2), dpi=100)
    ax0.set_ylim([0, 1])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    ax0.errorbar(x, y, yerr=error, fmt='-o')
    if title is not None:
        ax0.set_title(title)

    if save_loc is not None:
        fig.savefig('result_plots/{}.png'.format(save_loc), dpi=fig.dpi)


def multiline(xs, ys, c, ax=None, **kwargs):
    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    # Note: adding a collection doesn't autoscale xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def plot_multiline(x, Y, x_label, y_label, x_ticks, y_ticks, map_type='slope', title=None, save_loc=None):
    n_lines = len(Y)
    colors = np.arange(n_lines)

    mapping = []
    if map_type == 'slope':
        for y in Y:
            try:
                slope, intercept = np.polyfit(x, y, 1)
            except TypeError:
                y = np.array(y, dtype=np.float)
                slope, intercept = np.polyfit(x, y, 1)

            mapping.append(slope)
    elif map_type == 'delta':
        for y in Y:
            delta = y[0] - y[-1]
            mapping.append(delta)
    X = [x] * n_lines

    fig, ax = plt.subplots(figsize=(5, 2), dpi=100)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    lc = multiline(X, Y, mapping, cmap='binary', lw=1)

    axcb = fig.colorbar(lc)
    axcb.set_label('Slope')

    ax = plt.gca()
    ax.set_ylim([0, 1])
    if title is not None:
        ax.set_title(title)

    # adding random performance
    ax.plot(
        x, [1 / ((i + 1) * 2) for i in range(len(x))], color='red',
        linestyle='-', linewidth=0.5
    )

    if save_loc is not None:
        fig.savefig('result_plots/multiline_{}.png'.format(save_loc), dpi=fig.dpi)


def plot_train_loss(train_losses, save_loc):
    plt.figure(figsize=(5, 2), dpi=100)
    plt.title("Training Loss")
    plt.plot(train_losses, label="train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('{}.png'.format(save_loc), dpi=100)


def plot_val_accuracy(val_losses, save_loc):
    plt.figure(figsize=(5, 2), dpi=100)
    plt.title("Validation Accuracy")
    for idx, val_loss in enumerate(val_losses):
        plt.plot(val_loss, label='val accuracy [{}]'.format(idx))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig('{}.png'.format(save_loc), dpi=100)