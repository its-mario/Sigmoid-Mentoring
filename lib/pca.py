import numpy as np
from matplotlib import pyplot as plt

markers = {0: "o", 1: "^", 2: "*", 3: "#"}
colors = {0: "red", 1: "blue", 2: "green", 3: "yellow"}


def plot_pca(
        x_axes: "np.array",
        y_axes: "np.array",
        target: "np.array",
        title="",
):
    # combine all axes

    plot_array = np.stack((x_axes, y_axes, target), axis=1)  # axis = 1 transposes the stacked matrix

    # plot everything
    fig, ax = plt.subplots()
    ax.set_title(title)

    for row in plot_array:
        x_axes = row[0]
        y_axes = row[1]

        m = markers[int(row[2])]
        c = colors[int(row[2])]
        ax.scatter(x_axes, y_axes, marker=m, color=c)
