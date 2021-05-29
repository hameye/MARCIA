#########################################################################
#    This file is part of MARCIA developed at the University of Lorraine
#    by the GeoRessources Laboratory. MARCIA helps building masks and
#    clusters based on the knowledge of the user about the sample.
#
#    MARCIA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MARCIA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MARCIA.  If not, see <https://www.gnu.org/licenses/>
#
#    Author = Hadrien Meyer
#    Contact = jean.cauzid@univ-lorraine.fr
#    Copyright (C) 2019, 2020 H. Meyer, University of Lorraine
#
#########################################################################


######################################################
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import hyperspy.api as hs
from matplotlib.colors import ListedColormap
from PIL import Image
hs.preferences.GUIs.warn_if_guis_are_missing = False
hs.preferences.save()
######################################################
__author__ = "Hadrien Meyer"
__organization__ = "ENSG - UMR GeoRessources N°7359 - Université de Lorraine"
__email__ = "meyerhadrien96@gmail.com"
__date__ = "March, 2020"

plt.rcParams['image.cmap'] = 'cividis'


class Mask:

    """Class that allows to do a mineralogical classification of a
    sample provided the elemental data and a spreadsheet containing
    the elemental information needed for each mineral.
    The classification indicates the minerals, the percentage of each,
    the percentage of pixels that are classified more than once.
    It also indicates the percentage of the number of pixel that are not
    indexed.
    It also enables to extract a binaray image per mineral in order
    to use it as a mask onto the datacube (.rpl file) to facilitate
    quantitative analysis by the software.

    The spreadsheet can contain elemental and ratios querries and also a line
    for the color querries.

    """

    def __init__(self,
                 prefix: str,
                 suffix: str,
                 table: str,
                 normalization: bool = True):
        """
        Initialization of the class.
        Extraction of the suffix of the file in order to know how to treat it.
        Indication of the presence of a scalebar if the file is an image.
        Indication of the will to have normalized information of the
        element :intensity between 0 and 100.
                If so, values in the spreadsheet are specified between 0 and 1.
                If not, values in the spreadsheet are specified in number
                of counts in the spectrum.
                If the file is an image, the normalization is automatic.

        Parameters
        ----------
        prefix : str
            Common name to all data files (eg: lead_ore_)
        suffix : {'.bmp', '.tif', '.jpg','.txt','.rpl'}
            Type of the data file
        table : str
            Name of spreadsheet containing thresholds (eg: Mask.xlsx)
        normalization : bool, optional
            Indicate if data are normalized or not, only valid
            for non images data files.
        """
        self.prefix = prefix
        self.suffix = suffix
        self.normalization = normalization
        self.table_name = table
        self.colors = None

    def create_mineral_mask(self):
        """
        Create a 2D array that associate each pixel to a mask
        by assigning a value to each pixel. It also creates a
        dictionnary containing the relative proportion of a value
        compared to others.
        """
        # Creation of proportion dictionnary
        proportion = {}

        # Initialization of 2D array
        array = np.zeros((self.data_cube.shape[0], self.data_cube.shape[1]))

        # Convert the array to nan values
        array[np.isfinite(array)] = np.nan

        # Loop over the mask to check pixels that are assigned more than once
        for indice in range(len(self.Minerals)):
            array[(np.isfinite(self.mineral_cube[:, :, indice])) & (
                np.nansum(self.mineral_cube, axis=2) == 1)] = indice
        array[np.where(np.nansum(self.mineral_cube, axis=2) > 1)
              ] = len(self.Minerals) + 1
        for indice in range(len(self.Minerals)):
            proportion[indice] = np.where(array == indice)[
                0].shape[0] / np.sum(np.isfinite(array)) * 100
        return array

    def _create_mineral_mask_and_prop(self):
        """
        Create a 2D array that associate each pixel to a mask
        by assigning a value to each pixel. It also creates a
        dictionnary containing the relative proportion of a value
        compared to others.
        """
        # Creation of proportion dictionnary
        proportion = {}

        # Initialization of 2D array
        array = np.zeros((self.data_cube.shape[0], self.data_cube.shape[1]))

        # Convert the array to nan values
        array[np.isfinite(array)] = np.nan

        # Loop over the mask to check pixels that are assigned more than once
        for indice in range(len(self.Minerals)):
            array[(np.isfinite(self.mineral_cube[:, :, indice])) & (
                np.nansum(self.mineral_cube, axis=2) == 1)] = indice
        array[np.where(np.nansum(self.mineral_cube, axis=2) > 1)
              ] = len(self.Minerals) + 1
        for indice in range(len(self.Minerals)):
            proportion[indice] = np.where(array == indice)[
                0].shape[0] / np.sum(np.isfinite(array)) * 100
        return array, proportion

    def plot_mineral_mask(self):
        """
        For mineralogy purposes, valid only if all masks are minerals
        Plot all the mask onto one picture in order to visualize
        the classification. Each pixel correspond to only one mineral
        at the time, if not, it is classified as "mixed".

        """
        fig = plt.figure()

        array, proportion = self._create_mineral_mask_and_prop()

        # First plot to generate random colors
        im = plt.imshow(array, cmap='Paired')

        # Store finite values for later purpose
        finite_values_array = array[np.isfinite(array)]

        # Check if mixed pixels, need to add one more value
        if np.nansum(
                self.mineral_cube,
                axis=2).max() > 1:
            values = np.arange(len(self.Minerals) + 1)
        else:
            values = np.arange(len(self.Minerals))

        colors = [im.cmap(im.norm(value)) for value in values]
        plt.close()
        # Test if colors where specify in the table
        if self.colors:
            # If true, specified values are replaced
            for value in self.colors:
                colors[value] = self.colors[value]

        # Generating the new colormap
        new_colormap = ListedColormap(colors)

        # Open new figure
        fig = plt.figure()
        im = plt.imshow(array,
                        cmap=new_colormap,
                        vmin=values.min(),
                        vmax=values.max())

        # create a patch for every color
        # If true, there are mixed pixels: need to add a patch of mixte
        if np.nanmax(array) > len(self.Minerals):
            patches = [
                mpatches.Patch(
                    color=colors[np.where(
                        values == int(i))[0][0]],
                    label="{} : {} %".format(
                        self.Minerals[int(i)],
                        str(
                            round(
                                proportion[
                                    int(i)],
                                2)))) for i in values[
                    :-1] if round(
                    proportion[
                        int(i)], 2) > 0]

            patches.append(mpatches.Patch(
                color=colors[-1],
                label="{} : {} %".format(
                    'Misclassified',
                    str(round(
                        np.where(array == np.nanmax(
                            array))[0].shape[0] / np.sum(
                            np.isfinite(array)) * 100,
                        2)))))

        # If False, just add patches of corresponding masks
        else:
            patches = [
                mpatches.Patch(
                    color=colors[
                        np.where(
                            values == int(i))[0][0]],
                    label="{} : {} %".format(
                        self.Minerals[
                            int(i)], str(
                            round(
                                proportion[
                                    int(i)], 2)))) for i in values[:] if round(
                                        proportion[
                                            int(i)], 2) > 0]

        # Finally add a patch to specify proporty of non-classified pixel
        # Two reasons : images is bigger than sample or misclassification
        patches.append(
            mpatches.Patch(
                color='white',
                label="{} : {} %".format(
                    'Not classified', str(
                        round(
                            (self.data_cube.shape[0]
                                * self.data_cube.shape[1]
                                - len(finite_values_array))
                            / (self.data_cube.shape[0]
                               * self.data_cube.shape[1])
                            * 100, 2)))))

        # Add patches to the legend
        plt.legend(handles=patches,
                   bbox_to_anchor=(1.05, 1),
                   loc=2,
                   borderaxespad=0.)
        plt.grid(True)
        plt.title("Mineralogical classification - " + self.prefix[:-1])
        plt.tight_layout()
        plt.show()

    def get_masked_element(self, element: str, mineral: str):
        """
        Plot the elemental map and the histogram
        associated only in a specific mask.

        Parameters
        ----------
        element : str
            Name of the wanted element (eg: 'Fe')
        mineral : str
            Name of the wanted mask (eg: 'Galene')

        """
        # Conversion of given string indices to integer indices of the cubes
        element = list(self.Elements.values()).index(str(element))
        mineral = list(self.Minerals.values()).index(str(mineral))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax = axes.ravel()
        Anan = self.data_cube[:, :, element][np.isfinite(
            self.data_cube[:, :, element])]
        array = self.data_cube[:, :, element]
        array[np.isnan(self.mineral_cube[:, :, mineral])] = 0
        im = ax[0].imshow(array)
        ax[0].grid()
        ax[0].set_title("Carte élémentaire de {} masquéé par {}".format(
            self.Elements[element], self.Minerals[mineral]))
        fig.colorbar(im, ax=ax[0])
        plt.ylim(0, np.max(Anan))
        sns.distplot(Anan, kde=False, ax=axes[1], hist_kws={
                     'range': (0.0, np.max(Anan))}, vertical=True)
        ax[1].set_xscale('log')
        ax[1].set_title("Histograme d'intensité : " + self.Elements[element])
        fig.tight_layout()
        plt.show()

    def cube_masking_keep(self, mineral: str):
        """
        Recreates a raw datacube containing data only
        in the wanted mask.

        Parameters
        ----------
        mineral : str
            Name of the wanted mask (eg: 'Galene')

        """
        # Conversion of given string indices to integer indice of the cube
        mineral = list(self.Minerals.values()).index(str(mineral))
        cube = hs.load(self.prefix[:-1] + ".rpl",
                       signal_type="EDS_SEM",
                       lazy=True)
        array = np.asarray(cube)
        array[np.isnan(self.mineral_cube[:, :, mineral])] = 0
        cube = hs.signals.Signal1D(array)
        cube.save(self.prefix[:-1] + '_mask_kept_' +
                  self.Minerals[mineral] + ".rpl",
                  encoding='utf8')
        f = open(self.prefix[:-1] + ".rpl", "r")
        output = open(self.prefix[:-1] + '_mask_kept_' +
                      self.Minerals[mineral] + ".rpl",
                      'w')
        output.write(f.read())
        f.close()
        output.close()

    def cube_masking_remove(self, mineral: str):
        """
        Recreates a raw datacube containing all the
        data without the mask not wanted.

        Parameters
        ----------
        mineral : str
            Name of the wanted mask (eg: 'Galene')

        """
        # Conversion of given string indices to integer indice of the cube
        if mineral == 'mixed':
            a = self.create_mineral_mask()[0]
            mixed = np.where(a < np.nanmax(a), np.nan, a)
            cube = hs.load(self.prefix[:-1] + ".rpl",
                           signal_type="EDS_SEM",
                           lazy=True)
            array = np.asarray(cube)
            array[np.isfinite(mixed)] = 0
            cube = hs.signals.Signal1D(array)
            cube.save(self.prefix[:-1] + '_mask_removed_mixed' + ".rpl",
                      encoding='utf8')
            f = open(self.prefix[:-1] + ".rpl", "r")
            output = open(self.prefix[:-1] + '_mask_removed_mixed' + ".rpl",
                          'w')
            output.write(f.read())
            f.close()
            output.close()

        elif mineral == 'not indexed':
            a = self.create_mineral_mask()[0]
            nan = np.where(np.isnan(a), 0, a)
            cube = hs.load(self.prefix[:-1] + ".rpl",
                           signal_type="EDS_SEM",
                           lazy=True)
            array = np.asarray(cube)
            array[np.isfinite(nan)] = 0
            cube = hs.signals.Signal1D(array)
            cube.save(self.prefix[:-1] + '_mask_removed_nan' +
                      ".rpl",
                      encoding='utf8')
            f = open(self.prefix[:-1] + ".rpl", "r")
            output = open(self.prefix[:-1] + '_mask_removed_nan' +
                          + ".rpl",
                          'w')
            output.write(f.read())
            f.close()
            output.close()
        else:
            mineral = list(self.Minerals.values()).index(str(mineral))
            cube = hs.load(self.prefix[:-1] + ".rpl",
                           signal_type="EDS_SEM",
                           lazy=True)
            array = np.asarray(cube)
            array[np.isfinite(self.mineral_cube[:, :, mineral])] = 0
            cube = hs.signals.Signal1D(array)
            cube.save(self.prefix[:-1] + '_mask_removed_' +
                      self.Minerals[mineral] + ".rpl",
                      encoding='utf8')
            f = open(self.prefix[:-1] + ".rpl", "r")
            output = open(self.prefix[:-1] + '_mask_removed_' +
                          self.Minerals[mineral] + ".rpl",
                          'w')
            output.write(f.read())
            f.close()
            output.close()

    def save_mask_spectrum(self, mask: str):
        """Save the mean spectrum of a given mask as a txt file
        First column is channel
        Second column is counts

        Parameters
        ----------
        mask : str
            Name of the wanted mask (eg: 'Galene')

        """

        mineral = list(self.Minerals.values()).index(
            str(mask))
        cube = hs.load(self.prefix[:-1] + ".rpl",
                       signal_type="EDS_SEM",
                       lazy=True)
        array = np.asarray(cube)
        array[np.isnan(self.mineral_cube[:, :, mineral])] = 0
        cube = hs.signals.Signal1D(array)
        spectrum = cube.sum().data
        d = {'Counts': spectrum}
        dataframe = pd.DataFrame(data=d)
        dataframe.index.name = 'channel'
        dataframe.to_csv(mask + '_mean_spectrum.txt')
