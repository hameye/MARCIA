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
__organization__ = "ENSG - Université de Lorraine"
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
    for the color querries."""

    def __init__(
            self,
            prefix: str,
            suffix: str,
            table: str,
            Scale: bool = False,
            Normalization: bool = True):
        """Initialization of the class.
        Extraction of the suffix of the file in order to know how to treat it.
        Indication of the presence of a scalebar if the file is an image.
        Indication of the will to have normalized information of the
        element :intensity between 0 and 100.
                If so, values in the spreadsheet are specified between 0 and 1.
                If not, values in the spreadsheet are specified in number
                of counts in the spectrum.
                If the file is an image, the normalization is automatic.
        """
        self.prefix_ = prefix
        self.suffix_ = suffix
        self.scale_ = Scale
        self.Normalization_ = Normalization
        self.table_name = table

    def load_table(self):
        """ Load the spreadsheet into the programm.
        Verification if information of colors are required for
        the classification. Colors are also specified in the spreadsheet.
        """

        # Check if table is csv/txt or xlsx
        if self.table_name.split('.')[-1] in ('csv', 'txt'):
            self.table_ = pd.read_csv(self.table_name)
        elif self.table_name.split('.')[-1] in ('xls'):
            self.table_ = pd.read_excel(self.table_name)
        else:
            raise Exception("Please provide valid Table format.\
                Valid data types are: png, .bmp, .tif, .txt or .rpl ")

        # Check if table has specific colors for the masks
        if self.table_['Element'].str.contains('ouleur').any():
            indice = np.where(
                self.table_['Element'].str.contains('ouleur'))[0][0]

            # Creation of dictionnary containing the colors
            self.Couleurs_ = {}
            for coul in range(1, self.table_.iloc[indice].shape[0]):
                if isinstance(self.table_.iloc[indice][coul], str):
                    self.Couleurs_[coul - 1] = self.table_.iloc[indice][coul]

            # For simplicity in the process, color column is then removed
            self.table_ = self.table_.drop([indice])

    def datacube_creation(self):
        """ Function that creates a 3D array (X and Y are the dimensions of the
        sample and Z dimension is the number of elements/emission lines taken
        into account for the classification)
        It stacks the information contained in the elemental files given ranked
        according to the spreasheet ranking.
        If the normalization is asked or if the elemental map is an image,
        the data in the array are between 0 and 100.
        If there is a scalebar, the corresponding pixels are non assigned.

        Three types of elemental files are accepted :
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
        if self.suffix_ in ('.bmp', '.tif', '.jpg'):

            # Creation of element names dictionnary
            self.Elements_ = {}

            # Read the first image to know the dimensions
            test_image = np.linalg.norm(
                imread(
                    self.prefix_ +
                    self.table_.iloc[0][0] +
                    self.suffix_),
                axis=2)
            self.datacube_ = np.zeros(
                (test_image.shape[0],
                 test_image.shape[1], self.table_.shape[0]))
            test_image[:, :] = 0

            # Loop over elements in the table
            for element in range(self.table_.shape[0]):
                self.Elements_[element] = self.table_.iloc[element]['Element']

                # Check if the element is not a ratio of two elements
                if '/' not in self.table_.iloc[element]['Element']:

                    # Load of the  RGB image and normalization to one component
                    self.datacube_[
                        :,
                        :,
                        element] = np.linalg.norm(
                        imread(
                            self.prefix_ +
                            self.table_.iloc[element][0] +
                            self.suffix_),
                        axis=2)

                    # If there is a scale, scale is usually white, making it
                    # the highest value of the map, adding all maps and then
                    # remove values above a threshold allows to remove scale of
                    # calculation
                    if self.scale_:
                        test_image += self.datacube_[:, :, element]

                # If the element is actually a ratio of two elements
                else:

                    # Load of the two images
                    image_over = imread(
                        self.prefix_ +
                        self.table_['Element'][element].split('/')[0] +
                        self.suffix_)
                    image_under = imread(
                        self.prefix_ +
                        self.table_['Element'][element].split('/')[1] +
                        self.suffix_)

                    # Normalization of the two images
                    image_over_grey = np.linalg.norm(image_over, axis=2)
                    image_under_grey = np.linalg.norm(image_under, axis=2)

                    # Set 0 values to 0.01 in denominator image in order to
                    # avoid division by 0
                    image_under_grey[image_under_grey == 0.] = 0.01
                    self.datacube_[
                        :, :, element] = image_over_grey / image_under_grey

            # Arbitrary threshold fixed to 3000, but scale option might be
            # removed
            if self.scale_:
                for i in range(len(self.Elements_)):
                    self.datacube_[:, :, i][test_image > 3000] = np.nan

            # Normalization over 100 to every element of the cube
            for i in range(len(self.Elements_)):
                self.datacube_[:, :, i] = self.datacube_[
                    :, :, i] / np.nanmax(self.datacube_[:, :, i]) * 100

        # Check if data are textfiles consisting of raw count data per pixel
        # per energy domain
        elif self.suffix_ == '.txt':
            self.Elements_ = {}
            # Read the first image to know the dimensions
            test_image = np.loadtxt(
                self.prefix_ +
                self.table_.iloc[0][0] +
                self.suffix_,
                delimiter=';')
            self.datacube_ = np.zeros(
                (test_image.shape[0],
                 test_image.shape[1], self.table_.shape[0]))
            test_image[:, :] = 0

            # Loop over elements in the table
            for element in range(self.table_.shape[0]):
                self.Elements_[element] = self.table_.iloc[element]['Element']

                # Check if the element is not a ratio of two elements
                if '/' not in self.table_.iloc[element]['Element']:
                    # Load of the data count per element
                    self.datacube_[
                        :,
                        :,
                        element] = np.loadtxt(
                        self.prefix_ +
                        self.table_.iloc[element][0] +
                        self.suffix_,
                        delimiter=';')
                    if self.echelle_:
                        test_image += self.datacube_[:, :, element]

                # If the element is actually a ratio of two elements
                else:
                    image_over_grey = np.loadtxt(
                        self.prefix_ +
                        self.table_['Element'][element].split('/')[0] +
                        self.suffix_,
                        delimiter=';')
                    image_under_grey = np.loadtxt(
                        self.prefix_ +
                        self.table_['Element'][element].split('/')[1] +
                        self.suffix_,
                        delimiter=';')

                    self.datacube_[
                        :, :, element] = image_over_grey / image_under_grey
            # If user wants to see normalized over 100 data
            # This option makes impossible intensity comparison over element
            if self.Normalization_:
                for i in range(len(self.Elements_)):
                    self.datacube_[:, :, i] = self.datacube_[
                        :, :, i] / np.nanmax(self.datacube_[:, :, i]) * 100

        # Check if data are .rpl file, that is complete datacube
        # Load of the file using HyperSpy library
        elif self.suffix_ == '.rpl':
            cube = hs.load(self.prefix_[:-1] + ".rpl",
                           signal_type="EDS_SEM", lazy=True)
            cube.axes_manager[-1].name = 'E'
            cube.axes_manager['E'].units = 'keV'
            cube.axes_manager['E'].scale = 0.01
            cube.axes_manager['E'].offset = -0.97
            self.Elements_ = {}
            test = np.linalg.norm(
                imread(
                    self.prefix_ +
                    self.table_.iloc[0][0] +
                    self.suffix_),
                axis=2)
            self.datacube_ = np.zeros(
                (test.shape[0], test.shape[1], self.table_.shape[0]))
            test[:, :] = 0

            for element in range(self.table_.shape[0]):
                self.Elements_[element] = self.table_.iloc[element]['Element']

                if '/' not in self.table_.iloc[element]['Element']:
                    cube.set_elements([self.table_.iloc[element]['Element']])
                    array = cube.get_lines_intensity()
                    self.datacube_[:, :, element] = np.asarray(array[0])

                    if self.echelle_:
                        test += self.datacube_[:, :, element]

                else:
                    image_over = imread(
                        self.prefix_ +
                        self.table_['Element'][element].split('/')[0] +
                        self.suffix_)
                    image_under = imread(
                        self.prefix_ +
                        self.table_['Element'][element].split('/')[1] +
                        self.suffix_)
                    image_over_grey = np.linalg.norm(image_over, axis=2)
                    image_under_grey = np.linalg.norm(image_under, axis=2)
                    image_under_grey[image_under_grey == 0.] = 0.001
                    self.datacube_[
                        :, :, element] = image_over_grey / image_under_grey

            if self.Normalisation_:
                for i in range(len(self.Elements_)):
                    self.datacube_[:, :, i] = self.datacube_[
                        :, :, i] / np.nanmax(self.datacube_[:, :, i]) * 100

        # Raise Exception to provide valide data type
        else:
            raise Exception("Please input valid data type. Valid data types\
                are: png, .bmp, .tif, .txt or .rpl ")

    def mineralcube_creation(self):
        """Function that creates a 3D numpy array (X and Y are the dimensions
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

        self.Minerals_ = {}
        self.mineralcube_ = np.zeros(
            (self.datacube_.shape[0],
             self.datacube_.shape[1],
             self.table_.shape[1] - 1))

        for mask in range(1, self.table_.shape[1]):
            name = self.table_.columns[mask]
            self.Minerals_[mask - 1] = name
            str_table = self.table_[name].astype('str', copy=True)
            index_str = np.where(self.table_[name].notnull())[0]
            mask_i_str = np.zeros(
                (self.datacube_.shape[0],
                 self.datacube_.shape[1],
                 index_str.shape[0]))

            for k in range(index_str.shape[0]):
                mask_i_str[:, :, k] = self.datacube_[:, :, index_str[k]]

                if len(str_table[index_str[k]].split('-')) == 1:
                    threshold_min = float(
                        str_table[index_str[k]].split('-')[0])
                    threshold_max = None

                else:
                    threshold_min = float(
                        str_table[index_str[k]].split('-')[0])
                    threshold_max = float(
                        str_table[index_str[k]].split('-')[1])

                if self.Normalisation_:
                    mask_i_str[:, :, k][mask_i_str[
                        :, :, k] < threshold_min * np.nanmax(
                            mask_i_str[:, :, k])] = np.nan
                    if threshold_max:
                        mask_i_str[:, :, k][mask_i_str[
                            :, :, k] > threshold_max * np.nanmax(
                                mask_i_str[:, :, k])] = np.nan

                    mask_i_str[np.isfinite(mask_i_str)] = 1
                else:
                    mask_i_str[:, :, k][mask_i_str[:, :, k]
                                        < threshold_min] = np.nan
                    if threshold_max:
                        mask_i_str[:, :, k][mask_i_str[:, :, k]
                                            > threshold_max] = np.nan

                    mask_i_str[np.isfinite(mask_i_str)] = 1

            mask_i_str = np.nansum(mask_i_str, axis=2)
            mask_i_str[mask_i_str < np.max(mask_i_str)] = np.nan
            self.mineralcube_[:, :, mask - 1] = mask_i_str / \
                np.nanmax(mask_i_str)

    def get_mask(self, indice: str):
        """Function that plots the mineral mask wanted
        Input is the index of the mineral in the 3D array (cube).
        """

        indice = list(self.Minerals_.values()).index(str(indice))
        fig = plt.figure()
        plt.imshow(self.mineralcube_[:, :, indice])
        plt.title(self.Minerals_[indice])
        plt.grid()
        plt.show()

    def save_mask(self, indice: str):
        """Function that saves the mineral mask wanted as a .tif file.
        Input is the index of the mineral in the 3D array (cube).
        """
        indice = list(self.Minerals_.values()).index(str(indice))
        plt.imshow(self.mineralcube_[:, :, indice])
        plt.title(self.Minerals_[indice])
        plt.savefig('Mask_' + self.Minerals_[indice] + '.tif')
        plt.close()

    def get_hist(self, indice: str):
        """Function that plots the elemental map and the corresponding
        hitogram of intensity
        Input is the index of the element in the 3D array
        Useful function in order to set the threshold in the spreadsheet."""
        indice = list(self.Elements_.values()).index(str(indice))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax = axes.ravel()
        Anan = self.datacube_[:, :, indice][np.isfinite(
            self.datacube_[:, :, indice])]
        im = ax[0].imshow(self.datacube_[:, :, indice])
        ax[0].grid()
        ax[0].set_title("Carte élémentaire : " + self.Elements_[indice])
        fig.colorbar(im, ax=ax[0])
        plt.ylim(0, np.max(Anan))
        sns.distplot(Anan, kde=False, ax=axes[1], hist_kws={
                     'range': (0.0, np.max(Anan))}, vertical=True)
        ax[1].set_xscale('log')
        ax[1].set_title("Histograme d'intensité : " + self.Elements_[indice])
        fig.tight_layout()
        plt.show()

    def create_mineral_mask(self):
        proportion = {}
        array = np.zeros((self.datacube_.shape[0], self.datacube_.shape[1]))
        array[np.isfinite(array)] = np.nan
        for indice in range(len(self.Minerals_)):
            array[(np.isfinite(self.mineralcube_[:, :, indice])) & (
                np.nansum(self.mineralcube_, axis=2) == 1)] = indice
        array[np.where(np.nansum(self.mineralcube_, axis=2) > 1)
              ] = len(self.Minerals_) + 1
        for indice in range(len(self.Minerals_)):
            proportion[indice] = np.where(array == indice)[
                0].shape[0] / np.sum(np.isfinite(array)) * 100

    def plot_mineral_mask(self):
        fig = plt.figure()
        proportion = {}
        array = np.zeros((self.datacube_.shape[0], self.datacube_.shape[1]))
        array[np.isfinite(array)] = np.nan
        for indice in range(len(self.Minerals_)):
            array[(np.isfinite(self.mineralcube_[:, :, indice])) & (
                np.nansum(self.mineralcube_, axis=2) == 1)] = indice
        array[np.where(np.nansum(self.mineralcube_, axis=2) > 1)
              ] = len(self.Minerals_) + 1
        for indice in range(len(self.Minerals_)):
            proportion[indice] = np.where(array == indice)[
                0].shape[0] / np.sum(np.isfinite(array)) * 100
        im = plt.imshow(array, cmap='Paired')
        arrray = array[np.isfinite(array)]
        values = np.unique(arrray.ravel())
        colors = [im.cmap(im.norm(value)) for value in values]

        try:
            for value in self.Couleurs_:
                colors[value] = self.Couleurs_[value]
        except:
            pass

        self.newcmp = ListedColormap(colors)
        plt.close()
        fig = plt.figure()
        im = plt.imshow(array, cmap=self.newcmp)
        # create a patch (proxy artist) for every color
        if np.nanmax(array) > len(self.Minerals_):
            patches = [
                mpatches.Patch(
                    color=colors[
                        np.where(
                            values == int(i))[0][0]], label="{} : {} %".format(
                        self.Minerals_[
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
        else:
            patches = [
                mpatches.Patch(
                    color=colors[
                        np.where(
                            values == int(i))[0][0]], label="{} : {} %".format(
                        self.Minerals_[
                            int(i)], str(
                            round(
                                proportion[
                                    int(i)], 2)))) for i in values[:] if round(
                                        proportion[
                                            int(i)], 2) > 0]
        patches.append(
            mpatches.Patch(
                color='white', label="{} : {} %".format(
                    'Non indexés', str(
                        round(
                            (self.datacube_.shape[0] * self.datacube_.shape[1] - len(arrray)) / (
                                self.datacube_.shape[0] * self.datacube_.shape[1]) * 100, 2)))))
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(
            1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)
        plt.title("Classificiation minéralogique - " + self.prefix_[:-1])
        plt.tight_layout()
        plt.show()

    def get_masked_element(self, element: str, mineral: str):
        """ Function that plots the elemental map and the histogram
        associated only in a specific mask.
        """
        element = list(self.Elements_.values()).index(str(element))
        mineral = list(self.Minerals_.values()).index(str(mineral))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax = axes.ravel()
        Anan = self.datacube_[:, :, element][np.isfinite(
            self.datacube_[:, :, element])]
        array = self.datacube_[:, :, element]
        array[np.isnan(self.mineralcube_[:, :, mineral])] = 0
        im = ax[0].imshow(array)
        ax[0].grid()
        ax[0].set_title("Carte élémentaire de {} masquéé par {}".format(
            self.Elements_[element], self.Minerals_[mineral]))
        fig.colorbar(im, ax=ax[0])
        plt.ylim(0, np.max(Anan))
        sns.distplot(Anan, kde=False, ax=axes[1], hist_kws={
                     'range': (0.0, np.max(Anan))}, vertical=True)
        ax[1].set_xscale('log')
        ax[1].set_title("Histograme d'intensité : " + self.Elements_[element])
        fig.tight_layout()
        plt.show()

    def cube_masking_keep(self, mineral: str):
        """Function that recreates a raw datacube containing data only
        in the wanted mask.
        """
        mineral = list(self.Minerals_.values()).index(str(mineral))
        cube = hs.load(self.prefix_[:-1] + ".rpl",
                       signal_type="EDS_SEM", lazy=True)
        array = np.asarray(cube)
        array[np.isnan(self.mineralcube_[:, :, mineral])] = 0
        cube = hs.signals.Signal1D(array)
        cube.save(self.prefix_[:-1] + '_mask_kept_' +
                  self.Minerals_[mineral] + ".rpl", encoding='utf8')
        f = open(self.prefix_[:-1] + ".rpl", "r")
        output = open(self.prefix_[:-1] + '_mask_kept_' +
                      self.Minerals_[mineral] + ".rpl", 'w')
        output.write(f.read())
        f.close()
        output.close()

    def cube_masking_remove(self, mineral: str):
        """Function that recreates a raw datacube containing all the
        data without the mask not wanted.
        """
        mineral = list(self.Minerals_.values()).index(str(mineral))
        cube = hs.load(self.prefix_[:-1] + ".rpl",
                       signal_type="EDS_SEM", lazy=True)
        array = np.asarray(cube)
        array[np.isfinite(self.mineralcube_[:, :, mineral])] = 0
        cube = hs.signals.Signal1D(array)
        cube.save(self.prefix_[:-1] + '_mask_removed_' +
                  self.Minerals_[mineral] + ".rpl", encoding='utf8')
        f = open(self.prefix_[:-1] + ".rpl", "r")
        output = open(self.prefix_[:-1] + '_mask_removed_' +
                      self.Minerals_[mineral] + ".rpl", 'w')
        output.write(f.read())
        f.close()
        output.close()

    def get_biplot(self, indicex: str, indicey: str):
        """Function that plots one element against another one in a
        scatter plot
        Input is the indexes of each of the two element in the 3D array
        Useful function in order to see elemental ratios and some
        elemental thresholds."""
        indicex = list(self.Elements_.values()).index(str(indicex))
        indicey = list(self.Elements_.values()).index(str(indicey))
        fig, axes = plt.subplots()
        Valuesx = self.datacube_[
            :, :, indicex][np.isfinite(self.datacube_[:, :, indicex])]
        Valuesy = self.datacube_[
            :, :, indicey][np.isfinite(self.datacube_[:, :, indicey])]
        plt.xlim(0, np.max(Valuesx))
        plt.ylim(0, np.max(Valuesy))
        plt.xlabel(str(self.Elements_[indicex]))
        plt.ylabel(str(self.Elements_[indicey]))
        sns.scatterplot(x=Valuesx, y=Valuesy, alpha=0.3, marker="+")
        fig.tight_layout()
        plt.show()

    def get_triplot(self, indicex: str, indicey: str, indicez: str):
        """Function that plots one element against another one in a scatter
        plot
        Input is the indexes of each of the two element in the 3D array
        Useful function in order to see elemental ratios and some elemental
        thresholds."""
        indicex = list(self.Elements_.values()).index(str(indicex))
        indicey = list(self.Elements_.values()).index(str(indicey))
        indicez = list(self.Elements_.values()).index(str(indicez))
        fig, axes = plt.subplots()
        Valuesx = self.datacube_[
            :, :, indicex][np.isfinite(self.datacube_[:, :, indicex])]
        Valuesy = self.datacube_[
            :, :, indicey][np.isfinite(self.datacube_[:, :, indicey])]
        Valuesz = self.datacube_[
            :, :, indicez][np.isfinite(self.datacube_[:, :, indicez])]
        plt.xlim(0, np.max(Valuesx))
        plt.ylim(0, np.max(Valuesy))
        plt.xlabel(str(self.Elements_[indicex]))
        plt.ylabel(str(self.Elements_[indicey]))
        plt.title(str(self.Elements_[indicez]))
        sns.scatterplot(x=Valuesx, y=Valuesy, hue=Valuesz,
                        alpha=0.3, marker="+")
        fig.tight_layout()
        plt.show()
