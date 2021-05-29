import glob
from typing import List

import hyperspy.api as hs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

from ..core.datacube import DataCube, HyperCube, MultiCube


def load(prefix: str, suffix: str, normalization: bool = True) -> DataCube:
    """
    Create a 3D array (X and Y are the dimensions of the
    sample and Z dimension is the number of elements/emission lines taken
    into account for the classification)
    It stacks the information contained in the elemental files given ranked
    according to the spreasheet ranking.
    If the normalization is asked or if the elemental map is an image,
    the data in the array are between 0 and 100.
    If there is a scalebar, the corresponding pixels are non assigned.

    Three types of elemental files are accepted
    -------------------------------------------
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

    """
    # Extract file list, needed if images
    file_list = glob.glob(f'{prefix}*{suffix}')
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
        print('Measuring tools defaut parameters are BLABLABAL')
        return load_hypermap(prefix,
                             suffix,
                             name='E',
                             units='keV',
                             scale=0.01, offset=-0.97)

    # Raise Exception to provide valide data type
    else:
        raise Exception(f"{prefix} invalid data type. "
                        f"Valid data types are: "
                        f".png, .bmp, .tif, .txt or .rpl ")


def load_images(file_list: List[str], prefix: str, suffix: str) -> MultiCube:
    # Set automatic normalization to True
    normalization = True
    elements = {}
    element_list = [elt.replace(f'{prefix}_', '').replace(
        f'{suffix}', '') for elt in file_list]

    # Creation of element names dictionnary

    # Read the first image to know the dimensions
    test_image = np.linalg.norm(
        np.array(Image.open(file_list[0])),
        axis=2)

    data_cube = np.zeros(
        (test_image.shape[0],
            test_image.shape[1],
            len(file_list)))

    test_image[:, :] = 0

    # Loop over elements in the table
    for element in range(len(file_list)):
        elements[element] = element_list[element]

        # Load of the  RGB image and normalization to one component
        data_cube[:, :, element] = np.linalg.norm(
            np.array(Image.open(file_list[element])),
            axis=2)

        data_cube[:, :, element] = data_cube[
            :, :, element] / np.nanmax(data_cube[:, :, element]) * 100

    return MultiCube(data_cube, elements, prefix, suffix, normalization=True)


def load_textfile(file_list: List[str],
                  prefix: str,
                  suffix: str,
                  normalization: bool = True) -> MultiCube:
    elements = {}
    element_list = [elt.replace(f'{prefix}_', '').replace(
        f'{suffix}', '') for elt in file_list]

    # Read the first image to know the dimensions
    test_image = np.loadtxt(file_list[0],
                            delimiter=';')

    data_cube = np.zeros(
        (test_image.shape[0],
            test_image.shape[1],
            len(file_list)))
    test_image[:, :] = 0

    # Loop over elements in the table
    for element in range(len(file_list)):
        elements[element] = element_list[element]

        # Load of the data count per element
        data_cube[:, :, element] = np.loadtxt(
            file_list[element],
            delimiter=';')

        # If user wants to see normalized over 100 data
        # This option makes impossible intensity comparison over element
        if normalization:
            data_cube[:, :, element] = data_cube[
                :, :, element] / np.nanmax(data_cube[:, :, element]) * 100

    return MultiCube(data_cube, elements, prefix, suffix, normalization)


def load_hypermap(prefix: str,
                  suffix: str,
                  name: str,
                  units: str,
                  scale: float,
                  offset: float) -> HyperCube:
    cube = hs.load(f'{prefix}{suffix}',
                   signal_type="EDS_SEM",
                   lazy=True)
    cube.axes_manager[-1].name = 'E'
    cube.axes_manager['E'].units = 'keV'
    cube.axes_manager['E'].scale = 0.01
    cube.axes_manager['E'].offset = -0.97

    return HyperCube(cube, prefix, suffix)

def save(datacube:DataCube, indice: str, raw: bool = False):
    """
    Save the mineral mask wanted as a .tif file.
    Input is the index of the mineral in the 3D array (cube).

    Parameters
    ----------
    indice : str
        Name of the wanted element (eg: 'Fe')

    """
    indice = list(datacube.elements.values()).index(str(indice))
    if not raw:
        # Conversion of given string indices to integer indice of the cube
        plt.imshow(datacube.datacube[:, :, indice])
        plt.title(datacube.elements[indice])
        plt.savefig('Mask_' + datacube.elements[indice] + '.tif')
        plt.close()
    else:
        test_array = (
            datacube.datacube[
                :,
                :,
                indice] * 255).astype(
            np.uint8)
        image = Image.fromarray(test_array)
        image.save('Mask_' + datacube.elements[indice] + '.tif')
