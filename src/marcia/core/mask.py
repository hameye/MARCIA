from typing import List

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

class Mask:
    """Mask Fitter Class
    Mask class is made for manual fitting of DataCube.

    Attributes:
        table (pd.DataFrame): Table to extract thresholds 
            for manual fitting.
        colors (List): Color List for plotting consistency.
        filename (str): Filename of table.
    """

    def __init__(self,
                 table: pd.DataFrame,
                 colors: List,
                 filename: str):
        self._table = table
        self._colors = colors
        self._filename = filename

    @property
    def table(self) -> pd.DataFrame:
        """Returns pandas table.

        Returns:
            DataFrame of thresholds values.
        """
        return self._table

    @table.setter
    def table(self,
              new_table: pd.DataFrame):
        self._table = new_table

    @property
    def masks(self) -> List:
        """Returns list of mask names.

        Returns:
            Names of user defined masks.
        """
        return self.table.columns.to_list()

    @property
    def filename(self) -> str:
        """Returns Filename.

        Returns:
            Name of file.
        """
        return self._filename

    @property
    def colors(self) -> List:
        """Returns colors list.

        Returns:
            User defined colors for each mask.
        """
        # Check if table has specific colors for the masks
        return self._colors

    def set_value(self,
                  mask: str,
                  element: str,
                  value: str):
        """Change value of threshold for
        a defined mask.

        Args:
            element (str): Element from Mask to be change updated.
            mask (str): Mask to be updated.
            value (str): Value to be set.
        """
        self._table[mask][element] = value

    def get_value(self,
                  element: str,
                  mask: str):
        """Get value of threshold for a defined mask.

        Args:
            element (str): Element from Mask to be change updated.
            mask (str):  Mask to be updated.

        Returns:
            [type]: [description]
        """
        return self._table[mask][element]
