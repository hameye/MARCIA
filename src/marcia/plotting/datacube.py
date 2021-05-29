import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from ..core.datacube import DataCube

mpl.rcParams['image.cmap'] = 'cividis'


def hist(datacube: DataCube, indice: str):
    """
    Plot the elemental map on the left side an
    the corresponding hitogram of intensity on the right side
    Input is the index of the element in the 3D array
    Useful function in order to set the threshold in the spreadsheet.

    Parameters
    ----------
    indice : str
        Name of the wanted element (eg: 'Fe')

    """
    # Conversion of given string indices to integer indice of the cube
    indice = list(datacube.elements.values()).index(str(indice))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes.ravel()

    # Keep only finite values
    finite_data = datacube.datacube[:, :, indice][np.isfinite(
        datacube.datacube[:, :, indice])]
    im = ax[0].imshow(datacube.datacube[:, :, indice])
    ax[0].grid()
    ax[0].set_title("Carte élémentaire : " + datacube.elements[indice])
    fig.colorbar(im, ax=ax[0])
    plt.ylim(0, np.max(finite_data))
    sns.distplot(finite_data,
                 kde=False,
                 ax=axes[1],
                 hist_kws={'range': (0.0, np.max(finite_data))},
                 vertical=True)

    # Logarithm scale because background has a lof ot points and flatten
    # interesting information if linear
    ax[1].set_xscale('log')
    ax[1].set_title("Histograme d'intensité : " + datacube.elements[indice])
    fig.tight_layout()
    plt.show()


def plot(datacube: DataCube, indice: str):
    """
    Plot the mineral mask wanted
    Input is the index of the mineral in the 3D array (cube).

    Parameters
    ----------
    indice : str
        Name of the wanted mask (eg: 'Galene')

    """
    # Conversion of given string indices to integer indice of the cube
    indice = list(datacube.elements.values()).index(str(indice))
    fig = plt.figure()
    plt.imshow(datacube.datacube[:, :, indice])
    plt.title(datacube.elements[indice])
    plt.grid()
    plt.show()


def biplot(datacube: DataCube, indicex: str, indicey: str):
    """
    Plot one element against another one in a scatter plot
    Input is the indexes of each of the two element in the 3D array
    Useful function in order to see elemental ratios and some
    elemental thresholds.

    Parameters
    ----------
    indicex : str
        Name of the wanted element on x axis (eg: 'Fe')
    indicey : str
        Name of the wanted element on y axis (eg: 'Pb')

    """
    # Conversion of given string indices to integer indices of the cubes
    indicex = list(datacube.Elements.values()).index(str(indicex))
    indicey = list(datacube.Elements.values()).index(str(indicey))
    fig, axes = plt.subplots()

    # Number of points limited to 100,000 for computationnal time

    Valuesx = datacube.data_cube[
        :, :, indicex][np.isfinite(datacube.data_cube[:, :, indicex])]
    Valuesy = datacube.data_cube[
        :, :, indicey][np.isfinite(datacube.data_cube[:, :, indicey])]

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


def triplot(datacube: DataCube, indicex: str, indicey: str, indicez: str):
    """
    Plot one element against another one in a scatter plot
    Input is the indexes of each of the two element in the 3D array
    Useful function in order to see elemental ratios and some elemental
    thresholds.

    Parameters
    ----------
    indicex : str
        Name of the wanted element on x axis(eg: 'Fe')
    indicey : str
        Name of the wanted element on y axis (eg: 'Pb')
    indicez : str
        Name of the wanted element on colorscale (eg: 'Cu')

    """
    # Conversion of given string indices to integer indices of the cubes
    indicex = list(datacube.Elements.values()).index(str(indicex))
    indicey = list(datacube.Elements.values()).index(str(indicey))
    indicez = list(datacube.Elements.values()).index(str(indicez))
    fig, axes = plt.subplots()
    Valuesx = datacube.data_cube[
        :, :, indicex][np.isfinite(datacube.data_cube[:, :, indicex])]
    Valuesy = datacube.data_cube[
        :, :, indicey][np.isfinite(datacube.data_cube[:, :, indicey])]
    Valuesz = datacube.data_cube[
        :, :, indicez][np.isfinite(datacube.data_cube[:, :, indicez])]

    data = {'x': Valuesx, 'y': Valuesy, 'z': Valuesz}
    df = pd.DataFrame(data)

    if len(df) > 100000:
        print('Number of points limited to 100000')
        df = df.sample(n=100000)
        df = df.reset_index().drop(columns=['index'])

    plt.xlim(0, np.max(Valuesx))
    plt.ylim(0, np.max(Valuesy))

    plt.title(str(datacube.Elements[indicez]))
    sns.scatterplot(x=df.x,
                    y=df.y,
                    hue=df.z,
                    alpha=0.3,
                    marker="+")
    plt.xlabel(str(datacube.Elements[indicex]))
    plt.ylabel(str(datacube.Elements[indicey]))
    fig.tight_layout()
    plt.show()
