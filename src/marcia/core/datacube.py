from typing import Dict

import numpy as np


class DataCube:
    def __init__(self,
                 datacube: np.ndarray,
                 prefix: str,
                 suffix: str):
        self._datacube = datacube
        self._prefix = prefix
        self._suffix = suffix

    @property
    def datacube(self):
        return self._datacube

    @property
    def prefix(self):
        return self._prefix

    @property
    def suffix(self):
        return self._suffix

    def plot(self, element: str):
        pass


class MultiCube(DataCube):
    def __init__(self,
                 data_cube: np.ndarray,
                 elements: Dict,
                 prefix: str,
                 suffix: str,
                 normalization: bool = True
                 ):
        self._elements = elements
        self._normalization = normalization

        super().__init__(data_cube, prefix, suffix)

    @property
    def elements(self):
        return self._elements

    @property
    def normalization(self):
        return self._normalization


class MineralCube(MultiCube):
    def __init__(self, data_cube, elements, prefix, suffix, normalization=True, colors=None):
        self._colors = colors
        super().__init__(data_cube, elements, prefix, suffix, normalization)

    def map(self):
        """
        Create a 2D array that associate each pixel to a mask
        by assigning a value to each pixel. It also creates a
        dictionnary containing the relative proportion of a value
        compared to others.
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
    def __init__(self, data_cube, prefix, suffix):
        super().__init__(data_cube, prefix, suffix)
