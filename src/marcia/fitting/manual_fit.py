import numpy as np

from ..core.datacube import MineralCube, MultiCube
from ..core.mask import Mask


def mineralcube_creation(cube: MultiCube,
                         mask: Mask) -> MineralCube:
    """
    Create a 3D numpy array (X and Y are the dimensions
    of the sample and Z dimension is the number of minerals wanted for
    the classification).
    Minerals are defined by the columns in the mask table. The 2D
    array created per mineral depends on the threshold specified in the
    mask table.
    If one value is given, it corresponds to the minimum threshold to
    be in the mineral.
    If two values separated by a dash, it corresponds to the range of
    values for this element to be in the mineral (threshold are
    exclusive).

    Each mineral array is binary with 1 where the pixel is in the
    mineral and NaN (not assigned value) where the pixel is not in the mineral.

    The function also creates a dictionnary containing the Z position
    of the minerals in the 3D array created.

    Args:
        cube: Marcia DataCube Object to be classify.
        mask: Marcia Mask Object containing thresholds
            for manual classification.

    Returns:
        Marcia Mineralube Object containing binary values.

    """
    # Creation of mineral/mask names dictionnary
    Minerals = {}

    # Intializing data cube
    mineral_cube = np.zeros((cube.datacube.shape[0],
                             cube.datacube.shape[1],
                             mask.table.shape[1]))

    # Loop over each mask in order to fill the cube and dictionnary
    for element in range(mask.table.shape[1]):

        # Extract name of the mask
        name = mask.table.columns[element]

        # Fill the dictionnary, the key being an integer index
        Minerals[element] = name

        # Values are convert to string in order to facilitate later split
        str_table = mask.table[name].astype('str', copy=True)

        # Keeping indices of elements that are used in a mask
        index_str = np.where(mask.table[name].notnull())[0]

        # Initializing intermediate 3D array
        mask_i_str = np.zeros((cube.datacube.shape[0],
                               cube.datacube.shape[1],
                               index_str.shape[0]))

        # Loop over elements of the mask
        for k in range(index_str.shape[0]):
            mask_i_str[:, :, k] = cube.datacube[:, :, index_str[k]]

            # If only one value in the table: it corresponds to minimum
            # threshold
            if len(str_table[index_str[k]].split('-')) == 1:
                threshold_min = float(str_table[index_str[k]].split('-')[0])
                threshold_max = None

            # If more thant one value (should be 2): it corresponds to the
            # range of accepted values
            else:
                threshold_min = float(str_table[index_str[k]].split('-')[0])
                threshold_max = float(str_table[index_str[k]].split('-')[1])

            # If the value are normalized, the threshold is between 0 and
            # 1: need to compare to maximum value
            if cube.normalization:
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
        mineral_cube[:, :, element] = mask_i_str / np.nanmax(mask_i_str)

    # Dividing by sum of all minerals for each in order to detail misclassified
    mineral_cube = mineral_cube

    # Create mineralcube object
    datacube = MineralCube(mineral_cube,
                           Minerals,
                           cube.prefix,
                           cube.suffix,
                           True,
                           mask.colors)

    return datacube
