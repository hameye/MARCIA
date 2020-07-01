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

    Attributes
    ----------
    colors : dict
        Description
    data_cube : TYPE
        Description
    Elements : dict
        Description
    mineral_cube : TYPE
        Description
    Minerals : dict
        Description
    newcmp : TYPE
        Description
    normalization : TYPE
        Description
    prefix : TYPE
        Description
    scale : TYPE
        Description
    suffix : TYPE
        Description
    table : TYPE
        Description
    table_name : TYPE
        Description
    """

    def __init__(self,
                 prefix: str,
                 suffix: str,
                 table: str,
                 normalization: bool = False):
        """Initialization of the class.
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
            Description
        suffix : str
            Description
        table : str
            Description
        scale : bool, optional
            Description
        normalization : bool, optional
            Description
        """
        self.prefix = prefix
        self.suffix = suffix
        self.normalization = normalization
        self.table_name = table

    def load_table(self):
        """Load the spreadsheet into the programm.
        Verification if information of colors are required for
        the classification. Colors are also specified in the spreadsheet.


        """

        # Check if table is csv/txt or xlsx
        if self.table_name.split('.')[-1] in ('csv', 'txt'):
            self.table = pd.read_csv(self.table_name)
        elif 'xls' in self.table_name.split('.')[-1]:
            self.table = pd.read_excel(self.table_name)
        else:
            raise Exception("Please provide valid Table format.\
                Valid data types are: png, .bmp, .tif, .txt or .rpl ")

        # Check if table has specific colors for the masks
        if self.table['Element'].str.contains('ouleur').any():
            indice = np.where(
                self.table['Element'].str.contains('ouleur'))[0][0]

            # Creation of dictionnary containing the colors
            self.colors = {}
            for coul in range(1, self.table.iloc[indice].shape[0]):
                if isinstance(self.table.iloc[indice][coul], str):
                    self.colors[coul - 1] = self.table.iloc[indice][coul]

            # For simplicity in the process, color column is then removed
            self.table = self.table.drop([indice])

    def datacube_creation(self):
        """Create a 3D array (X and Y are the dimensions of the
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
        values are the number of counts
        - Raw file (.rpl), wich is the datacube containing all the spectra
        for every pixel. The hyperspy library is used to extract the
        emission lines corresponding to the wanted elements.

        Textfiles and raw files can be normalized or not, the spreadsheet
        must be written according to that.

        The function also creates a dictionnary containing the Z position
        of the element in the 3D array created.

        2 class files created in that function.


        """

        # Check if the data files are images
        if self.suffix in ('.bmp', '.tif', '.jpg'):

            # Set automatic normalization to True
            self.normalization = True

            # Creation of element names dictionnary
            self.Elements = {}

            # Read the first image to know the dimensions
            test_image = np.linalg.norm(
                imread(
                    self.prefix +
                    self.table.iloc[0][0] +
                    self.suffix),
                axis=2)
            self.data_cube = np.zeros(
                (test_image.shape[0],
                 test_image.shape[1], self.table.shape[0]))
            test_image[:, :] = 0

            # Loop over elements in the table
            for element in range(self.table.shape[0]):
                self.Elements[element] = self.table.iloc[element]['Element']

                # Check if the element is not a ratio of two elements
                if '/' not in self.table.iloc[element]['Element']:

                    # Load of the  RGB image and normalization to one component
                    self.data_cube[
                        :,
                        :,
                        element] = np.linalg.norm(
                        imread(
                            self.prefix +
                            self.table.iloc[element][0] +
                            self.suffix),
                        axis=2)

                # If the element is actually a ratio of two elements
                else:

                    # Load of the two images
                    image_over = imread(
                        self.prefix +
                        self.table['Element'][element].split('/')[0] +
                        self.suffix)
                    image_under = imread(
                        self.prefix +
                        self.table['Element'][element].split('/')[1] +
                        self.suffix)

                    # Normalization of the two images
                    image_over_grey = np.linalg.norm(image_over, axis=2)
                    image_under_grey = np.linalg.norm(image_under, axis=2)

                    # Set 0 values to 0.01 in denominator image in order to
                    # avoid division by 0
                    image_under_grey[image_under_grey == 0.] = 0.01
                    self.data_cube[
                        :, :, element] = image_over_grey / image_under_grey

            # Normalization over 100 to every element of the cube
            for i in range(len(self.Elements)):
                self.data_cube[:, :, i] = self.data_cube[
                    :, :, i] / np.nanmax(self.data_cube[:, :, i]) * 100

        # Check if data are textfiles consisting of raw count data per pixel
        # per energy domain
        elif self.suffix == '.txt':
            self.Elements = {}
            # Read the first image to know the dimensions
            test_image = np.loadtxt(
                self.prefix +
                self.table.iloc[0][0] +
                self.suffix,
                delimiter=';')
            self.data_cube = np.zeros(
                (test_image.shape[0],
                 test_image.shape[1], self.table.shape[0]))
            test_image[:, :] = 0

            # Loop over elements in the table
            for element in range(self.table.shape[0]):
                self.Elements[element] = self.table.iloc[element]['Element']

                # Check if the element is not a ratio of two elements
                if '/' not in self.table.iloc[element]['Element']:
                    # Load of the data count per element
                    self.data_cube[
                        :,
                        :,
                        element] = np.loadtxt(
                        self.prefix +
                        self.table.iloc[element][0] +
                        self.suffix,
                        delimiter=';')

                # If the element is actually a ratio of two elements
                else:
                    image_over_grey = np.loadtxt(
                        self.prefix +
                        self.table['Element'][element].split('/')[0] +
                        self.suffix,
                        delimiter=';')
                    image_under_grey = np.loadtxt(
                        self.prefix +
                        self.table['Element'][element].split('/')[1] +
                        self.suffix,
                        delimiter=';')

                    self.data_cube[
                        :, :, element] = image_over_grey / image_under_grey
            # If user wants to see normalized over 100 data
            # This option makes impossible intensity comparison over element
            if self.normalization:
                for i in range(len(self.Elements)):
                    self.data_cube[:, :, i] = self.data_cube[
                        :, :, i] / np.nanmax(self.data_cube[:, :, i]) * 100

        # Check if data are .rpl file, that is complete datacube
        # Load of the file using HyperSpy library
        elif self.suffix == '.rpl':
            cube = hs.load(self.prefix[:-1] + ".rpl",
                           signal_type="EDS_SEM", lazy=True)
            cube.axes_manager[-1].name = 'E'
            cube.axes_manager['E'].units = 'keV'
            cube.axes_manager['E'].scale = 0.01
            cube.axes_manager['E'].offset = -0.97
            self.Elements = {}
            test = np.linalg.norm(
                imread(
                    self.prefix +
                    self.table.iloc[0][0] +
                    self.suffix),
                axis=2)
            self.data_cube = np.zeros(
                (test.shape[0], test.shape[1], self.table.shape[0]))
            test[:, :] = 0

            for element in range(self.table.shape[0]):
                self.Elements[element] = self.table.iloc[element]['Element']

                if '/' not in self.table.iloc[element]['Element']:
                    cube.set_elements([self.table.iloc[element]['Element']])
                    array = cube.get_lines_intensity()
                    self.data_cube[:, :, element] = np.asarray(array[0])

                    if self.echelle_:
                        test += self.data_cube[:, :, element]

                else:
                    image_over = imread(
                        self.prefix +
                        self.table['Element'][element].split('/')[0] +
                        self.suffix)
                    image_under = imread(
                        self.prefix +
                        self.table['Element'][element].split('/')[1] +
                        self.suffix)
                    image_over_grey = np.linalg.norm(image_over, axis=2)
                    image_under_grey = np.linalg.norm(image_under, axis=2)
                    image_under_grey[image_under_grey == 0.] = 0.001
                    self.data_cube[
                        :, :, element] = image_over_grey / image_under_grey

            if self.normalization:
                for i in range(len(self.Elements)):
                    self.data_cube[:, :, i] = self.data_cube[
                        :, :, i] / np.nanmax(self.data_cube[:, :, i]) * 100

        # Raise Exception to provide valide data type
        else:
            raise Exception("Please input valid data type. Valid data types\
                are: png, .bmp, .tif, .txt or .rpl ")

    def mineralcube_creation(self):
        """Create a 3D numpy array (X and Y are the dimensions
        of the sample and Z dimension is the number of minerals wanted for
        the classification).
        The minerals are defined by the columns in the spreadsheet. The 2D
        array create per mineral depends on the threshold specified in the
        spreadsheet.
        If one value is given, it corresponds to the minimum threshold to
        be in the mineral.
        If two values separated by a dash, it corresponds to the range of
        values for this element to be in the mineral.
        Given values are outside the range.

        Each mineral array is binary with 1 where the pixel is in the
        mineral and NaN (non assigned) where the pixel is not in the mineral.

        The function also creates a dictionnary containing the Z position
        of the minerals in the 3D array created.

        2 class files created in that function.

        """
        # Creation of mineral/mask names dictionnary
        self.Minerals = {}

        # Intializing data cube
        self.mineral_cube = np.zeros(
            (self.data_cube.shape[0],
             self.data_cube.shape[1],
             self.table.shape[1] - 1))

        # Loop over each mask in order to fill the cube and dictionnary
        for mask in range(1, self.table.shape[1]):

            # Extract name of the mask
            name = self.table.columns[mask]

            # Fill the dictionnary, the key being an integer index
            self.Minerals[mask - 1] = name

            # Values are convert to string in order to facilitate later split
            str_table = self.table[name].astype('str', copy=True)

            # Keeping indices of elements that are used in a mask
            index_str = np.where(self.table[name].notnull())[0]

            # Initializing intermediate 3D array
            mask_i_str = np.zeros(
                (self.data_cube.shape[0],
                 self.data_cube.shape[1],
                 index_str.shape[0]))

            # Loop over elements of the mask
            for k in range(index_str.shape[0]):
                mask_i_str[:, :, k] = self.data_cube[:, :, index_str[k]]

                # If only one value in the table: it corresponds to minimum
                # threshold
                if len(str_table[index_str[k]].split('-')) == 1:
                    threshold_min = float(
                        str_table[index_str[k]].split('-')[0])
                    threshold_max = None

                # If more thant one value (should be 2): it corresponds to the
                # range of accepted values
                else:
                    threshold_min = float(
                        str_table[index_str[k]].split('-')[0])
                    threshold_max = float(
                        str_table[index_str[k]].split('-')[1])

                # If the value are normalized, the threshold is between 0 and
                # 1: need to compare to maximum value
                if self.normalization:
                    mask_i_str[:, :, k][mask_i_str[
                        :, :, k] < threshold_min * np.nanmax(
                            mask_i_str[:, :, k])] = np.nan
                    if threshold_max:
                        mask_i_str[:, :, k][mask_i_str[
                            :, :, k] > threshold_max * np.nanmax(
                                mask_i_str[:, :, k])] = np.nan

                    # Values outside thresholds are nan, and valid values are
                    # set to 1
                    mask_i_str[np.isfinite(mask_i_str)] = 1

                # If not normalize, threshold is just the number of counts
                else:
                    mask_i_str[:, :, k][mask_i_str[:, :, k]
                                        < threshold_min] = np.nan
                    if threshold_max:
                        mask_i_str[:, :, k][mask_i_str[:, :, k]
                                            > threshold_max] = np.nan

                    # Values outside thresholds are nan, and valid values are
                    # set to 1
                    mask_i_str[np.isfinite(mask_i_str)] = 1

            # 3D array is stacked
            mask_i_str = np.nansum(mask_i_str, axis=2)

            # Mask correspond to maximum values: ones that satisfied all
            # conditions
            mask_i_str[mask_i_str < np.max(mask_i_str)] = np.nan

            # Mask cube 2D slice is filled with 1 where mask is true
            self.mineral_cube[:, :, mask - 1] = mask_i_str / \
                np.nanmax(mask_i_str)

    def get_mask(self, indice: str):
        """Plot the mineral mask wanted
        Input is the index of the mineral in the 3D array (cube).

        Parameters
        ----------
        indice : str
            Description

        """
        # Conversion of given string indices to integer indice of the cube
        indice = list(self.Minerals.values()).index(str(indice))
        fig = plt.figure()
        plt.imshow(self.mineral_cube[:, :, indice])
        plt.title(self.Minerals[indice])
        plt.grid()
        plt.show()

    def save_mask(self, indice: str):
        """Save the mineral mask wanted as a .tif file.
        Input is the index of the mineral in the 3D array (cube).

        Parameters
        ----------
        indice : str
            Description

        """
        # Conversion of given string indices to integer indice of the cube
        indice = list(self.Minerals.values()).index(str(indice))
        plt.imshow(self.mineral_cube[:, :, indice])
        plt.title(self.Minerals[indice])
        plt.savefig('Mask_' + self.Minerals[indice] + '.tif')
        plt.close()

    def get_hist(self, indice: str):
        """Plot the elemental map on the left side an
        the corresponding hitogram of intensity on the right side
        Input is the index of the element in the 3D array
        Useful function in order to set the threshold in the spreadsheet.

        Parameters
        ----------
        indice : str
            Description

        """
        # Conversion of given string indices to integer indice of the cube
        indice = list(self.Elements.values()).index(str(indice))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax = axes.ravel()

        # Keep only finite values
        Anan = self.data_cube[:, :, indice][np.isfinite(
            self.data_cube[:, :, indice])]
        im = ax[0].imshow(self.data_cube[:, :, indice])
        ax[0].grid()
        ax[0].set_title("Carte élémentaire : " + self.Elements[indice])
        fig.colorbar(im, ax=ax[0])
        plt.ylim(0, np.max(Anan))
        sns.distplot(Anan, kde=False, ax=axes[1], hist_kws={
                     'range': (0.0, np.max(Anan))}, vertical=True)

        # Logarithm scale because background has a lof ot points and flatten
        # interesting information if linear
        ax[1].set_xscale('log')
        ax[1].set_title("Histograme d'intensité : " + self.Elements[indice])
        fig.tight_layout()
        plt.show()

    def create_mineral_mask(self):
        """Create a 2D array that associate each pixel to a mask
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

    def plot_mineral_mask(self):
        """Summary
        """
        fig = plt.figure()

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

        # Loop over the mask to calculate proportion of each
        for indice in range(len(self.Minerals)):
            proportion[indice] = np.where(array == indice)[
                0].shape[0] / np.sum(np.isfinite(array)) * 100

        # First plot to generate random colors
        im = plt.imshow(array, cmap='Paired')
        arrray = array[np.isfinite(array)]

        if np.nansum(
                self.mineral_cube,
                axis=2).max() > 1:
            values = np.arange(len(self.Minerals) + 1)
        else:
            values = np.arange(len(self.Minerals))

        colors = [im.cmap(im.norm(value)) for value in values]

        # Test if colors where specify in the table
        try:

            # If true, specified values are replaced
            for value in self.colors:
                colors[value] = self.colors[value]
        except:
            pass

        # Generating the new colormap
        self.newcmp = ListedColormap(colors)
        plt.close()

        # Open new figure
        fig = plt.figure()
        im = plt.imshow(array,
                        cmap=self.newcmp,
                        vmin=values.min(),
                        vmax=values.max())

        # create a patch for every color
        # If true, there are mixed pixels: need to add a patch of mixte
        if np.nanmax(array) > len(self.Minerals):
            patches = [
                mpatches.Patch(
                    color=colors[
                        np.where(
                            values == int(i))[0][0]], label="{} : {} %".format(
                        self.Minerals[
                            int(i)], str(
                            round(
                                proportion[
                                    int(i)], 2)))) for i in values[
                                        :-1] if round(
                                            proportion[
                                                int(i)], 2) > 0]
            patches.append(mpatches.Patch(
                color=colors[-1], label="{} : {} %".format(
                    'Mixtes', str(round(
                        np.where(array == np.nanmax(
                            array))[0].shape[0] / np.sum(
                            np.isfinite(array)) * 100, 2)))))

        # If False, just add patches of corresponding masks
        else:
            patches = [
                mpatches.Patch(
                    color=colors[
                        np.where(
                            values == int(i))[0][0]], label="{} : {} %".format(
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
                color='white', label="{} : {} %".format(
                    'Non indexés', str(
                        round(
                            (self.data_cube.shape[
                                0] * self.data_cube.shape[1] - len(arrray)) / (
                                self.data_cube.shape[
                                    0] * self.data_cube.shape[1]) * 100, 2)))))

        # Add patches to the legend
        plt.legend(handles=patches, bbox_to_anchor=(
            1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)
        plt.title("Classification minéralogique - " + self.prefix[:-1])
        plt.tight_layout()
        plt.show()

    def get_masked_element(self, element: str, mineral: str):
        """Plot the elemental map and the histogram
        associated only in a specific mask.

        Parameters
        ----------
        element : str
            Description
        mineral : str
            Description

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
        """Recreates a raw datacube containing data only
        in the wanted mask.

        Parameters
        ----------
        mineral : str
            Description

        """
        # Conversion of given string indices to integer indice of the cube
        mineral = list(self.Minerals.values()).index(str(mineral))
        cube = hs.load(self.prefix[:-1] + ".rpl",
                       signal_type="EDS_SEM", lazy=True)
        array = np.asarray(cube)
        array[np.isnan(self.mineral_cube[:, :, mineral])] = 0
        cube = hs.signals.Signal1D(array)
        cube.save(self.prefix[:-1] + '_mask_kept_' +
                  self.Minerals[mineral] + ".rpl", encoding='utf8')
        f = open(self.prefix[:-1] + ".rpl", "r")
        output = open(self.prefix[:-1] + '_mask_kept_' +
                      self.Minerals[mineral] + ".rpl", 'w')
        output.write(f.read())
        f.close()
        output.close()

    def cube_masking_remove(self, mineral: str):
        """Recreates a raw datacube containing all the
        data without the mask not wanted.

        Parameters
        ----------
        mineral : str
            Description
        """
        # Conversion of given string indices to integer indice of the cube
        mineral = list(self.Minerals.values()).index(str(mineral))
        cube = hs.load(self.prefix[:-1] + ".rpl",
                       signal_type="EDS_SEM", lazy=True)
        array = np.asarray(cube)
        array[np.isfinite(self.mineral_cube[:, :, mineral])] = 0
        cube = hs.signals.Signal1D(array)
        cube.save(self.prefix[:-1] + '_mask_removed_' +
                  self.Minerals[mineral] + ".rpl", encoding='utf8')
        f = open(self.prefix[:-1] + ".rpl", "r")
        output = open(self.prefix[:-1] + '_mask_removed_' +
                      self.Minerals[mineral] + ".rpl", 'w')
        output.write(f.read())
        f.close()
        output.close()

    def get_biplot(self, indicex: str, indicey: str):
        """Plot one element against another one in a scatter plot
        Input is the indexes of each of the two element in the 3D array
        Useful function in order to see elemental ratios and some
        elemental thresholds.

        Parameters
        ----------
        indicex : str
            Description
        indicey : str
            Description
        """
        # Conversion of given string indices to integer indices of the cubes
        indicex = list(self.Elements.values()).index(str(indicex))
        indicey = list(self.Elements.values()).index(str(indicey))
        fig, axes = plt.subplots()
        Valuesx = self.data_cube[
            :, :, indicex][np.isfinite(self.data_cube[:, :, indicex])]
        Valuesy = self.data_cube[
            :, :, indicey][np.isfinite(self.data_cube[:, :, indicey])]
        plt.xlim(0, np.max(Valuesx))
        plt.ylim(0, np.max(Valuesy))
        plt.xlabel(str(self.Elements[indicex]))
        plt.ylabel(str(self.Elements[indicey]))
        sns.scatterplot(x=Valuesx, y=Valuesy, alpha=0.3, marker="+")
        fig.tight_layout()
        plt.show()

    def get_triplot(self, indicex: str, indicey: str, indicez: str):
        """Plot one element against another one in a scatter plot
        Input is the indexes of each of the two element in the 3D array
        Useful function in order to see elemental ratios and some elemental
        thresholds.

        Parameters
        ----------
        indicex : str
            Description
        indicey : str
            Description
        indicez : str
            Description
        """
        # Conversion of given string indices to integer indices of the cubes
        indicex = list(self.Elements.values()).index(str(indicex))
        indicey = list(self.Elements.values()).index(str(indicey))
        indicez = list(self.Elements.values()).index(str(indicez))
        fig, axes = plt.subplots()
        Valuesx = self.data_cube[
            :, :, indicex][np.isfinite(self.data_cube[:, :, indicex])]
        Valuesy = self.data_cube[
            :, :, indicey][np.isfinite(self.data_cube[:, :, indicey])]
        Valuesz = self.data_cube[
            :, :, indicez][np.isfinite(self.data_cube[:, :, indicez])]
        plt.xlim(0, np.max(Valuesx))
        plt.ylim(0, np.max(Valuesy))
        plt.xlabel(str(self.Elements[indicex]))
        plt.ylabel(str(self.Elements[indicey]))
        plt.title(str(self.Elements[indicez]))
        sns.scatterplot(x=Valuesx, y=Valuesy, hue=Valuesz,
                        alpha=0.3, marker="+")
        fig.tight_layout()
        plt.show()

    def get_mask_spectrum(self, mask: str):
        """Summary

        Parameters
        ----------
        mask : str
            Description
        """
        mineral = list(self.Minerals.values()).index(
            str(mask))
        cube = hs.load(self.prefix[:-1] + ".rpl",
                       signal_type="EDS_SEM", lazy=True)
        array = np.asarray(cube)
        array[np.isnan(self.mineral_cube[:, :, mineral])] = 0
        cube = hs.signals.Signal1D(array)
        spectrum = cube.sum().data
        dataframe = pd.DataFrame(spectrum, columns=['channel', 'counts'])
        #dataframe['Counts'] = spectrum
        dataframe.to_csv(mask + '_mean_spectrum.txt')
