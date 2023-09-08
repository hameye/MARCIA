import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

from ..core.datacube import DataCube, HyperCube, MineralCube, MultiCube

mpl.rcParams['image.cmap'] = 'cividis'


def hist_in_mask(
    data_cube: DataCube,
    mineral_cube: MineralCube,
    element: str,
    mineral: str
):
    """
    Plot elemental map and histogram
    associated only for a specific element in a mineralcube.

    Args:
        data_cube: DataCube to plot.
        mineral_cube: MineralCube to check if element in mineral.
        element:  Name of the wanted element (eg: 'Fe').
        mineral:  Name of the wanted mask (eg: 'Galene').

    Note:
        Does not work if datacube is HyperCube (yet).

    """
    # Conversion of given string indices to integer indices of the cubes
    element = list(data_cube.elements.values()).index(str(element))
    mineral = list(mineral_cube.elements.values()).index(str(mineral))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes.ravel()

    Anan = data_cube.datacube[:, :, element][np.isfinite(data_cube.datacube[:, :, element])]

    array = data_cube.datacube[:, :, element]
    array[np.isnan(mineral_cube.datacube[:, :, mineral])] = 0
    im = ax[0].imshow(array)
    ax[0].set_axis_off()
    ax[0].set_title(
        f"Carte élémentaire de {data_cube.elements[element]} masquéé par {mineral_cube.elements[mineral]}"
    )
    fig.colorbar(im, ax=ax[0])
    plt.ylim(0, np.max(Anan))
    sns.histplot(
        y=Anan,
        kde=False,
        ax=axes[1],
        line_kws={'range': (0.0, np.max(Anan))},
        bins=50,
        element="step"
    )
    ax[1].set_xscale('log')
    ax[1].set_title(f"Histograme d'intensité : {data_cube.elements[element]}")
    fig.tight_layout()
    plt.show()


def hist(
    datacube: DataCube,
    indice: str
):
    """Plot elemental map and histogram of a specific element.

    Args:
        datacube: Marcia DataCube Object.
        indice: Element to plot.
    """
    if type(datacube) == MultiCube:
        datacube_hist(datacube, indice)
    elif type(datacube) == MineralCube:
        mineralcube_hist(datacube, indice)
    elif type(datacube) == HyperCube:
        hypercube_hist(datacube, indice)
    else:
        raise NotImplementedError('Method not implemented yet ... ')


def datacube_hist(
    datacube: DataCube,
    indice: str
):
    """
    Plot the elemental map on the left side an
    the corresponding hitogram of intensity on the right side
    Input is the index of the element in the 3D array
    Useful function in order to set the threshold in the spreadsheet.

    Args:
        datacube: Marcia DataCube Object.
        indice: Element to plot.

    """
    # Conversion of given string indices to integer indice of the cube
    indice = list(datacube.elements.values()).index(str(indice))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes.ravel()

    # Keep only finite values
    finite_data = datacube.datacube[:, :, indice][np.isfinite(datacube.datacube[:, :, indice])]
    
    im = ax[0].imshow(datacube.datacube[:, :, indice])
    ax[0].set_axis_off()
    ax[0].set_title(f"Carte élémentaire : {datacube.elements[indice]}")
    fig.colorbar(im, ax=ax[0])
    plt.ylim(0, np.max(finite_data))
    sns.histplot(
        y=finite_data,
        kde=False,
        ax=axes[1],
        line_kws={'range': (0.0, np.max(finite_data))},
        bins=50,
        element="step"
    )

    # Logarithm scale because background has a lof ot points and flatten
    # interesting information if linear
    ax[1].set_xscale('log')
    ax[1].set_title(f"Histograme d'intensité : {datacube.elements[indice]}")
    fig.tight_layout()
    plt.show()


def mineralcube_hist(
    mineral_cube: MineralCube,
    indice: str
):
    """
    Plot the elemental map on the left side an
    the corresponding hitogram of probability to 
    be in the mineral on the right side.
    Input is the index of the element in the 3D array
    Useful function in order to set the threshold in the spreadsheet.

    Args:
        mineral_cube: Marcia DataCube Object.
        indice: Element to plot.

    """
    # Conversion of given string indices to integer indice of the cube
    indice = list(mineral_cube.elements.values()).index(str(indice))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes.ravel()

    data = (
        mineral_cube.datacube
        / np.nansum(mineral_cube.datacube, axis=2, keepdims=mineral_cube.datacube.shape[2])
    )

    # Keep only finite values
    finite_data = data[:, :, indice][np.isfinite(data[:, :, indice])]
    data = mineral_cube.datacube[:, :, indice].copy()
    data[np.isnan(data)] = 0
    im = ax[0].imshow(data)
    ax[0].set_axis_off()
    ax[0].set_title(f"Carte minérale : {mineral_cube.elements[indice]}")
    fig.colorbar(im, ax=ax[0])
    plt.ylim(0, np.max(finite_data))
    sns.histplot(
        y=finite_data,
        kde=False,
        ax=axes[1],
        line_kws={'range': (0.0, np.max(finite_data))},
        bins=50,
        element="step"
    )

    # Logarithm scale because background has a lof ot points and flatten
    # interesting information if linear
    ax[1].set_xscale('log')
    ax[1].set_title(f"Histograme de probabilité : {mineral_cube.elements[indice]}")
    fig.tight_layout()
    plt.show()


def hypercube_hist(
    datacube: HyperCube,
    elements: str
):
    datacube.datacube.set_elements([elements])
    t = np.asarray(datacube.datacube.get_lines_intensity()[0])
    if datacube.normalization:
        t = t / np.nanmax(t) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes.ravel()

    # Keep only finite values
    finite_data = t[np.isfinite(t)]
    im = ax[0].imshow(t)
    ax[0].set_axis_off()
    ax[0].set_title(f"Carte élémentaire : {elements}")
    fig.colorbar(im, ax=ax[0])
    plt.ylim(0, np.max(finite_data))
    sns.histplot(
        y=finite_data,
        kde=False,
        ax=axes[1],
        line_kws={'range': (0.0, np.max(finite_data))},
        bins=50,
        element="step"
    )

    # Logarithm scale because background has a lof ot points and flatten
    # interesting information if linear
    ax[1].set_xscale('log')
    ax[1].set_title(f"Histograme d'intensité : {elements}")
    fig.tight_layout()
    plt.show()


def plot(
    datacube: DataCube,
    indice: str
):
    """
    Plot the mineral mask wanted
    Input is the index of the mineral in the 3D array (cube).

    Args:
        datacube: Marcia DataCube Object.
        indice: Element to plot.

    """
    # Conversion of given string indices to integer indice of the cube
    indice = list(datacube.elements.values()).index(str(indice))
    fig = plt.figure()
    plt.imshow(datacube.datacube[:, :, indice])
    plt.title(datacube.elements[indice])
    plt.grid()
    plt.show()


def biplot(
    datacube: DataCube,
    indicex: str,
    indicey: str
):
    """
    Plot one element against another one in a scatter plot
    Input is the indexes of each of the two element in the 3D array
    Useful function in order to see elemental ratios and some
    elemental thresholds.

    Args:
        datacube: Marcia DataCube Object.
        indicex: Element to plot on x-axis.
        indicey: Element to plot on y-axis.

    """
    # Conversion of given string indices to integer indices of the cubes
    indicex = list(datacube.Elements.values()).index(str(indicex))
    indicey = list(datacube.Elements.values()).index(str(indicey))
    fig, axes = plt.subplots()

    # Number of points limited to 100,000 for computationnal time

    Valuesx = datacube.data_cube[:, :, indicex][np.isfinite(datacube.data_cube[:, :, indicex])]
    Valuesy = datacube.data_cube[:, :, indicey][np.isfinite(datacube.data_cube[:, :, indicey])]

    data = {'x': Valuesx, 'y': Valuesy}
    df = pd.DataFrame(data)

    # Limit number of samples to 100,000
    if len(df) > 100000:
        print('Number of points limited to 100000')
        df = df.sample(n=100000)
        df = df.reset_index().drop(columns=['index'])
    plt.xlim(0, np.max(Valuesx))
    plt.ylim(0, np.max(Valuesy))
    plt.xlabel(str(datacube.Elements[indicex]))
    plt.ylabel(str(datacube.Elements[indicey]))
    sns.scatterplot(x=df.x, y=df.y, alpha=0.3, marker="+")
    fig.tight_layout()
    plt.show()


def triplot(
    datacube: DataCube,
    indicex: str,
    indicey: str,
    indicez: str
):
    """
    Plot one element against another one in a scatter plot
    Input is the indexes of each of the two element in the 3D array
    Useful function in order to see elemental ratios and some elemental
    thresholds.

    Args:
        datacube: Marcia DataCube Object.
        indicex: Element to plot on x-axis.
        indicey: Element to plot on y-axis.
        indicez: Element to color data with.

    """
    # Conversion of given string indices to integer indices of the cubes
    indicex = list(datacube.Elements.values()).index(str(indicex))
    indicey = list(datacube.Elements.values()).index(str(indicey))
    indicez = list(datacube.Elements.values()).index(str(indicez))
    fig, axes = plt.subplots()
    Valuesx = datacube.data_cube[:, :, indicex][np.isfinite(datacube.data_cube[:, :, indicex])]
    Valuesy = datacube.data_cube[:, :, indicey][np.isfinite(datacube.data_cube[:, :, indicey])]
    Valuesz = datacube.data_cube[:, :, indicez][np.isfinite(datacube.data_cube[:, :, indicez])]

    data = {'x': Valuesx, 'y': Valuesy, 'z': Valuesz}
    df = pd.DataFrame(data)

    if len(df) > 100000:
        print('Number of points limited to 100000')
        df = df.sample(n=100000)
        df = df.reset_index().drop(columns=['index'])

    plt.xlim(0, np.max(Valuesx))
    plt.ylim(0, np.max(Valuesy))

    plt.title(str(datacube.Elements[indicez]))
    sns.scatterplot(
        x=df.x,
        y=df.y,
        hue=df.z,
        alpha=0.3,
        marker="+"
    )
    plt.xlabel(str(datacube.Elements[indicex]))
    plt.ylabel(str(datacube.Elements[indicey]))
    fig.tight_layout()
    plt.show()


def plot_minerals(datacube: MineralCube):
    """
    For mineralogy purposes, valid only if all masks are minerals
    Plot all the mask onto one picture in order to visualize
    the classification. Each pixel correspond to only one mineral
    at the time, if not, it is classified as "mixed".

    Args:
        datacube: Marcia MineralCube Object.

    """
    fig = plt.figure()

    array, proportion = datacube.map()

    # First plot to generate random colors
    im = plt.imshow(array, cmap='Paired')

    # Store finite values for later purpose
    finite_values_array = array[np.isfinite(array)]

    # Check if mixed pixels, need to add one more value
    if np.nansum(datacube.datacube, axis=2).max() > 1:
        values = np.arange(len(datacube.elements) + 1)
    else:
        values = np.arange(len(datacube.elements))

    colors = [im.cmap(im.norm(value)) for value in values]
    plt.close()

    # Test if colors where specify in the table
    if datacube._colors:
        # If true, specified values are replaced
        for value in range(len(datacube._colors)):
            if type(datacube._colors[value]) == str:
                colors[value] = datacube._colors[value]

    # Generating the new colormap
    new_colormap = ListedColormap(colors)

    # Open new figure
    fig = plt.figure(figsize=(10, 5))
    im = plt.imshow(
        array,
        cmap=new_colormap,
        vmin=values.min(),
        vmax=values.max()
    )

    # create a patch for every color
    # If true, there are mixed pixels: need to add a patch of mixte
    if np.nanmax(array) > len(datacube.elements):
        patches = [
            mpatches.Patch(
                color=colors[np.where(values == int(i))[0][0]],
                label=f"{datacube.elements[int(i)]}: {round(proportion[int(i)],2)} %"
            )
            for i in values[:-1]
            if round(proportion[int(i)], 2) > 0
        ]

        patches.append(
            mpatches.Patch(
                color=colors[-1],
                label=f"Misclassified: {round(np.where(array == np.nanmax(array))[0].shape[0] / np.sum(np.isfinite(array)) * 100,2)} %"
            )
        )

    # If False, just add patches of corresponding masks
    else:
        patches = [
            mpatches.Patch(
                color=colors[np.where(values == int(i))[0][0]],
                label=f"{datacube.elements[int(i)]}: {round(proportion[int(i)], 2)} %"
            )
            for i in values[:]
            if round(proportion[int(i)], 2) > 0
        ]

    # Finally add a patch to specify proporty of non-classified pixel
    # Two reasons : images is bigger than sample or misclassification
    patches.append(
        mpatches.Patch(
            color='white',
            label=f"Not classified: {round((datacube.datacube.shape[0]* datacube.datacube.shape[1]- len(finite_values_array))/ (datacube.datacube.shape[0]* datacube.datacube.shape[1])* 100, 2)} %"
        )
    )

    # Add patches to the legend
    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.
    )
    plt.title(f"Mineralogical classification - {datacube.prefix}")
    # plt.tight_layout()
    # plt.axis('off')
    plt.show()
