from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class DataCube(ABC):
    """Base Class Object for datacube.
    Considered as abstract parent class.

    Attributes:
        datacube (np.ndarray): multi-dimensionnal array containing data values.
        prefix (str): Common name between files.
        suffix (str): Extension name of the files.
    """
    @abstractmethod
    def __init__(self,
                 datacube: np.ndarray,
                 prefix: str,
                 suffix: str):
        """Init DataCube abstract class.

        """
        self._datacube = datacube
        self._prefix = prefix
        self._suffix = suffix

    @property
    def datacube(self) -> np.ndarray:
        """Return data values as a numpy array.

        Returns:
            3 dimensionnal numpy array.
        """
        return self._datacube

    @property
    def prefix(self) -> str:
        """Return prefix.

        Returns:
            prefix name.
        """
        return self._prefix

    @property
    def suffix(self) -> str:
        """Return suffix.

        Returns:
            suffix name.
        """
        return self._suffix


class MultiCube(DataCube):
    """MultiCube Object.
    MultiCube class is made for discrete datacube such
    as multispectral satellite data of stack of elemental
    images.

    Attributes:
        data_cube (np.ndarray): multi-dimensionnal array containing data values.
        elements (Dict): Dictionnary of channel names with associated index.
        prefix (str): Common name between files.
        suffix (str): Extension name of the files.
        normalization (bool): Indication of normalized data (0 to 100).
    """

    def __init__(self,
                 data_cube: np.ndarray,
                 elements: Dict,
                 prefix: str = None,
                 suffix: str = None,
                 normalization: bool = True):
        """Init MultiCube object.

        """
        self._elements = elements
        self._normalization = normalization

        super().__init__(data_cube, prefix, suffix)

    @property
    def elements(self) -> Dict:
        """Returns elements dictionnary.

        Returns:
            Keys are indices and values are element names.
        """
        return self._elements

    @property
    def normalization(self) -> bool:
        """Returns Normalization parameters.

        Returns:
            True if Normalized, False else.
        """
        return self._normalization


class MineralCube(MultiCube):
    """MineralCube Object.
    MultiCube object where values are binary (0 or 1).

    Class essentially made to create mask and make operations
    between data and masks.

    Attributes:
        data_cube (np.ndarrray): multi-dimensionnal array containing data values.
        elements (Dict): Dictionnary of channel names with associated index.
        prefix (str): Common name between files.
        suffix (str): Extension name of the files.
        normalization (bool, optionnal): Indication of normalized data (0 to 100).
        colors (List, optionnal): Color list provided for plotting consistency.
    """

    def __init__(self,
                 data_cube: np.ndarray,
                 elements: Dict,
                 prefix: str = None,
                 suffix: str = None,
                 normalization: bool = True,
                 colors: List = None):
        self._colors = colors
        super().__init__(data_cube, elements, prefix, suffix, normalization)

    @property
    def colors(self) -> List:
        """Returns colors lists.

        Returns:
            List of colors for plotting purposes.
        """
        return self._colors

    def map(self) -> Tuple[np.ndarray, Dict]:
        """
        Create a 2D array that associate each pixel to a mask
        by assigning a value to each pixel. It also creates a
        dictionnary containing the relative proportion of a value
        compared to others.

        Returns:
            Tuple containing 2D array to be used as an image, 
            and dictionnary containing proportion of each pixel value.
        """
        # Creation of proportion dictionnary
        proportion = {}

        # Initialization of 2D array
        array = np.zeros((self.datacube.shape[0], self.datacube.shape[1]))

        # Convert the array to nan values
        array[np.isfinite(array)] = np.nan

        # Loop over the mask to check pixels that are assigned more than once
        for indice in range(len(self.elements)):
            array[(np.isfinite(self.datacube[:, :, indice])) & (
                np.nansum(self.datacube, axis=2) == 1)] = indice

        array[np.where(np.nansum(self.datacube, axis=2) > 1)
              ] = len(self.elements) + 1

        for indice in range(len(self.elements)):
            proportion[indice] = np.where(array == indice)[
                0].shape[0] / np.sum(np.isfinite(array)) * 100

        return array, proportion


class HyperCube(DataCube):
    """HyperCube Object.
    HyperCube class is made for continuous datacube such
    as hyperspecral satellite data or hyperspectral datacube
    from MicroXRF or EPMA analyses.

    Attributes:
        data_cube (tbd): HyperSpy object.
        prefix (str): Filename.
        suffix (str): File extension.
    """

    def __init__(self,
                 data_cube,
                 prefix,
                 suffix,
                 normalization: bool = True):
        self._normalization = normalization
        super().__init__(data_cube, prefix, suffix)

    @property
    def normalization(self) -> bool:
        """Returns Normalization parameters.

        Returns:
            True if Normalized, False else.
        """
        return self._normalization

    def mineral_cube_creation_from_hps(self,
                                       elements_list: List):
        new_cube = np.zeros((self.datacube.axes_manager.shape[1],
                            self.datacube.axes_manager.shape[0],
                            len(elements_list)))
        for element in range(len(elements_list)):
            if '/' not in elements_list[element]:
                self.datacube.set_elements([elements_list[element]])
                array = self.datacube.get_lines_intensity()
                new_cube[:, :, element] = np.asarray(array[0])
            else:
                self.datacube.set_elements(
                    [elements_list[element].split('/')[0]])
                array = self.datacube.get_lines_intensity()
                image_over = np.asarray(array[0])
                self.datacube.set_elements(
                    [elements_list[element].split('/')[1]])
                array = self.datacube.get_lines_intensity()
                image_under = np.asarray(array[0])

                image_under[image_under == 0.] = 0.001
                new_cube[
                    :, :, element] = image_over / image_under

            if self.normalization:
                for i in range(len(elements_list)):
                    new_cube[:, :, i] = new_cube[
                        :, :, i] / np.nanmax(new_cube[:, :, i]) * 100
        elements = elements_list
        number = np.arange(len(elements_list))
        dictionnary = dict(zip(number, elements))

        return MultiCube(new_cube, dictionnary, None, None, False)
