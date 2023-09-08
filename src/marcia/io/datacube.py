import glob
import os.path
from typing import List

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ..core.datacube import DataCube, HyperCube, MultiCube


def load(
    prefix: str,
    suffix: str,
    normalization: bool = True,
    element_list: List[str] = None
):
    """
    Create a 3D array (X and Y are the dimensions of the
    sample and Z dimension is the number of elements/emission lines taken
    into account for the classification).

    Stack the information contained in the elemental files given ranked
    according to the spreasheet ranking.

    If normalization is asked or if the elemental map is an image file,
    data in the array are between 0 and 100.

    Three types of elemental files are accepted

    - Imges (.bmp of .tif), which are RGB files : each pixel contains 3
    values between 0 and 255. The rgb is put into greyscale calculated
    by the norm 2.

    - Textfile (.txt), which is already the elemental array where the
    values are the number of counts.

    - Raw file (.rpl), wich is the datacube containing all the spectra
    for every pixel. The hyperspy library is used to extract the
    emission lines corresponding to the wanted elements.

    Textfiles and raw files can be normalized or not, the spreadsheet
    must be written according to that.

    The function also creates a dictionnary containing the Z position
    of the element in the 3D array created.

    2 class files created in that function.

    Args:
        prefix: Common name between files.
        suffix: Extension name of the files.
        normalization: If data are normalized
            between 0 and 100.
        element_list: User can specify list 
            elements to be taken, or let the program
            finds all data in corresponding to given
            prefix name.

    Raises:
        TypeError: If given extension is not accepted by the program.
    """
    # Extract file list, needed if images
    if element_list is None:
        file_list = glob.glob(f'{prefix}*{suffix}')
        file_list.sort()
    else:
        file_list = [f"{prefix}_{elt}{suffix}" for elt in element_list]
        file_list.sort()

    # Check if the data files are images
    if suffix in ('.bmp', '.tif', '.jpg', '.png'):
        return load_images(file_list, prefix, suffix)

    # Check if data are textfiles consisting of raw count data per pixel
    # per energy
    elif suffix == '.txt':
        return load_textfile(file_list, prefix, suffix, normalization)

    # Check if data are .rpl file, that is complete datacube
    # Load of the file using HyperSpy library
    elif suffix == '.rpl':
        print(f"Measuring tools defaut parameters are scale = {0.01}, offset={0.97}")
        return load_hypermap(
            prefix,
            suffix,
            name='E',
            units='keV',
            scale=0.01, offset=-0.97
        )

    # Raise Exception to provide valide data type
    else:
        raise TypeError(
            f"{prefix} invalid data type. Valid data types are: .png, .jpg, .bmp, .tif, .txt or .rpl "
        )


def load_images(
    file_list: List[str],
    prefix: str,
    suffix: str
) -> MultiCube:
    """Images Data loader. 

    Low-level function for data loading.

    Args:
        file_list: List of images files to load.
        prefix: Common name between files.
        suffix: Extension name of the files.

    Returns:
        Marcia MultiCube Objects.

        MultiCube means data channels are discrete and not continuous.
    """
    # Set automatic normalization to True
    normalization = True

    # Creation of element names dictionnary
    elements = {}
    element_list = [elt.replace(f'{prefix}_', '').replace(f'{suffix}', '') for elt in file_list]

    # Read the first image to know the dimensions
    test_image = np.linalg.norm(np.array(Image.open(file_list[0])), axis=2)

    # Initialize multi-dimensionnal array
    data_cube = np.zeros((test_image.shape[0], test_image.shape[1], len(file_list)))

    # Loop over elements in the table
    for element in range(len(file_list)):
        elements[element] = element_list[element]

        # Load of the  RGB image and normalization to one component
        data_cube[:, :, element] = np.linalg.norm(np.array(Image.open(file_list[element])), axis=2)

        data_cube[:, :, element] = (
            data_cube[:, :, element]
            / np.nanmax(data_cube[:, :, element])
            * 100
        )

    return MultiCube(data_cube, elements, prefix, suffix, normalization=normalization)


def load_textfile(
    file_list: List[str],
    prefix: str,
    suffix: str,
    normalization: bool = True
) -> MultiCube:
    """Text Data Loader.

    Low-level function for data loading.

    Args:
        file_list: List of images files to load.
        prefix: Common name between files.
        suffix: Extension name of the files.
        normalization: If data are normalized
            between 0 and 100.

    Returns:
       Marcia MultiCube Objects.

       MultiCube means data channels are discrete and not continuous.
    """
    # Creation of element names dictionnary
    elements = {}
    element_list = [elt.replace(f'{prefix}_', '').replace(f'{suffix}', '') for elt in file_list]

    # Read the first image to know the dimensions
    test_image = np.loadtxt(file_list[0], delimiter=';')

    # Initialize multi-dimensionnal array
    data_cube = np.zeros((test_image.shape[0], test_image.shape[1], len(file_list)))

    # Loop over elements in the table
    for element in range(len(file_list)):
        elements[element] = element_list[element]

        # Load of the data count per element
        data_cube[:, :, element] = np.loadtxt(file_list[element], delimiter=';')

        # If user wants to see normalized over 100 data
        # This option makes impossible intensity comparison over element
        if normalization:
            data_cube[:, :, element] = (
                data_cube[:, :, element]
                / np.nanmax(data_cube[:, :, element])
                * 100
            )

    return MultiCube(data_cube, elements, prefix, suffix, normalization)


def load_hypermap(
    prefix: str,
    suffix: str,
    name: str = 'E',
    units: str = 'keV',
    scale: float = 0.01,
    offset: float = - 0.97
) -> HyperCube:
    """Raw HyperCube Data Loader.

    Args:
        prefix: Name of file.
        suffix: Extension name of the file.
        name: Physical measure of channel
        units: Unit of physical measure. Defaults to 'keV'.
        scale: Scale realationship betwen channel and physical measure.
        offset: Scale offset between channel and physcial measure.

    Returns:
        Marcia HyperCube Object.

        HyperCube means data channels are continuous.
    """
    cube = hs.load(f'{prefix}{suffix}', signal_type="EDS_SEM", lazy=True)
    cube.axes_manager[-1].name = name
    cube.axes_manager[name].units = units
    cube.axes_manager[name].scale = scale
    cube.axes_manager[name].offset = offset

    return HyperCube(cube, prefix, suffix)


def save(
    datacube: DataCube,
    element: str,
    raw: bool = False
):
    """
    Save specified element as a .tif image file.

    Args:
        datacube: Marcia DataCube Object.
        element: Name of the wanted element (eg: 'Fe')
        raw: If saved image has information background or not.

    """
    # Convert element name to index
    index = list(datacube.elements.values()).index(str(element))

    # If not raw, return image with matplotlibbackground
    if not raw:

        # Conversion of given string indices to integer indice of the cube
        plt.imshow(datacube.datacube[:, :, index])
        plt.title(datacube.elements[index])
        plt.savefig(f"mask_{datacube.elements[index]}.tif")
        plt.close()

    # Else, return only image withou background
    else:
        test_array = (datacube.datacube[:, :, index] * 255).astype(np.uint8)
        image = Image.fromarray(test_array)
        image.save(f"mask_{datacube.elements[index]}.tif")


def save_cube_inside_mask(data_cube: DataCube,
                          mineral: str):
    """
    Recreates raw datacube containing data only
    in the wanted element.

    Args:
        data_cube: Marcia DataCube Object.
        mineral: Name of the wanted element (eg: 'Galene')

    Raises:
        FileNotFoundError: If raw data cube is not in folder.

    """
    if not os.path.isfile(f"{data_cube.prefix}.raw") & os.path.isfile(f"{data_cube.prefix}.rpl"):
        raise FileNotFoundError("Raw Cube not found in files")

    # Conversion of given string indices to integer indice of the cube
    mineral_index = list(data_cube.elements.values()).index(str(mineral))

    cube = hs.load(f"{data_cube.prefix}.rpl", signal_type="EDS_SEM", lazy=True)

    array = np.asarray(cube)
    array[np.isnan(data_cube.datacube[:, :, mineral_index])] = 0
    cube = hs.signals.Signal1D(array)

    cube.save(
        f"{data_cube.prefix}_mask_kept_'{data_cube.elements[mineral_index]}.rpl",
        encoding='utf8'
    )

    f = open(f"{data_cube.prefix}.rpl", "r")
    output = open(f"{data_cube.prefix}_mask_kept_{data_cube.elements[mineral_index]}.rpl", 'w')
    output.write(f.read())
    f.close()
    output.close()


def save_cube_outside_mask(
    data_cube: DataCube,
    mineral: str
):
    """
    Recreates a raw datacube containing data that 
    are not in the specified element.

    Args:
        data_cube: Marcia DataCube Object.
        mineral: Name of the wanted element (eg: 'Galene')

    Raises:
        FileNotFoundError: If raw data cube is not in folder.

    """
    if not os.path.isfile(f"{data_cube.prefix}.raw") & os.path.isfile(f"{data_cube.prefix}.rpl"):
        raise FileNotFoundError("Raw Cube not found in files")

    # Conversion of given string indices to integer indice of the cube
    if mineral == 'mixed':
        a = data_cube.map()[0]
        mixed = np.where(a < np.nanmax(a), np.nan, a)
        cube = hs.load(f"{data_cube.prefix}.rpl", signal_type="EDS_SEM", lazy=True)

        array = np.asarray(cube)
        array[np.isfinite(mixed)] = 0
        cube = hs.signals.Signal1D(array)
        cube.save(f"{data_cube.prefix}_mask_removed_mixed.rpl", encoding='utf8')

        f = open(f"{data_cube.prefix}.rpl", "r")
        output = open(f"{data_cube.prefix}_mask_removed_mixed.rpl", 'w')
        output.write(f.read())
        f.close()
        output.close()

    elif mineral == 'not indexed':
        a = data_cube.map()[0]
        nan = np.where(np.isnan(a), 0, a)
        cube = hs.load(f"{data_cube.prefix}.rpl", signal_type="EDS_SEM", lazy=True)
        array = np.asarray(cube)
        array[np.isfinite(nan)] = 0
        cube = hs.signals.Signal1D(array)
        cube.save(f"{data_cube.prefix}_mask_removed_nan.rpl", encoding='utf8')
        f = open(f"{data_cube.prefix}.rpl", "r")
        output = open(f"{data_cube.prefix}_mask_removed_nan.rpl", 'w')
        output.write(f.read())
        f.close()
        output.close()
    else:
        mineral_index = list(data_cube.elements.values()).index(str(mineral))
        cube = hs.load(f"{data_cube.prefix}.rpl", signal_type="EDS_SEM", lazy=True)
        array = np.asarray(cube)
        array[np.isfinite(data_cube.datacube[:, :, mineral_index])] = 0
        cube = hs.signals.Signal1D(array)
        cube.save(
            f"{data_cube.prefix}_mask_removed_{data_cube.elements[mineral_index]}.rpl",
            encoding='utf8'
        )
        f = open(f"{data_cube.prefix}.rpl", "r")
        output = open(
            f"{data_cube.prefix}_mask_removed_'{data_cube.elements[mineral_index]}.rpl",
            'w'
        )
        output.write(f.read())
        f.close()
        output.close()


def save_mask_spectrum(
    data_cube: DataCube,
    mask: str
):
    """Save mean spectrum of a given mask as a .txt file.

    - First column is channel.
    - Second column is counts.

    Args:
        data_cube: Marcia DataCube Object.
        mask: Name of the wanted mask (eg: 'Galene').

    Raises:
        FileNotFoundError: If raw data cube is not in folder.

    """
    if not os.path.isfile(f"{data_cube.prefix}.raw") & os.path.isfile(f"{data_cube.prefix}.rpl"):
        raise FileNotFoundError("Raw Cube not found in files")

    mineral_index = list(data_cube.elements.values()).index(str(mask))
    cube = hs.load(f"{data_cube.prefix}.rpl", signal_type="EDS_SEM", lazy=True)

    array = np.asarray(cube)
    array[np.isnan(data_cube.datacube[:, :, mineral_index])] = 0
    cube = hs.signals.Signal1D(array)
    spectrum = cube.sum().data
    d = {'Counts': spectrum}
    dataframe = pd.DataFrame(data=d)
    dataframe.index.name = 'channel'
    dataframe.to_csv(f"{mask}_mean_spectrum.txt")
