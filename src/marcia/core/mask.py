from typing import List
from PIL.Image import new

import numpy as np
import pandas as pd



class Mask:
    def __init__(self, table: pd.DataFrame,colors:List, filename: str):
        self._table = table
        self._colors = colors
        self._filename = filename

    @property
    def table(self):
        return self._table

    @table.setter
    def table(self,new_table:pd.DataFrame):
        self._table = new_table

    @property
    def masks(self):
        return self.table.to_list()

    @property
    def filename(self):
        return self._filename
    
    @property
    def reload(self):
        if self.filename.split('.')[-1] in ('csv', 'txt'):
            table = pd.read_csv(self.filename, index_col='Element')

        elif 'xls' in self.filename.split('.')[-1]:
            table = pd.read_excel(self.filename, index_col='Element')
        self.table = table

    @property
    def colors(self) -> List:
        # Check if table has specific colors for the masks
        return self._colors
