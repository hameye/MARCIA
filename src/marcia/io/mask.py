import pandas as pd
import numpy as np
from ..core.mask import Mask


def load_mask(filename):
    """
    Load the spreadsheet into the programm.
    Verification if information of colors are required for
    the classification. Colors are also specified in the spreadsheet.

    """
    # Check if table is csv/txt or xlsx
    if filename.split('.')[-1] in ('csv', 'txt'):
        table = pd.read_csv(filename, index_col='Element')

    elif 'xls' in filename.split('.')[-1]:
        table = pd.read_excel(filename, index_col='Element')

    else:
        raise Exception(
            f"{filename.split('.')[-1]} "
            f"invalid Table format."
            f"Valid data types are: .csv, .txt, or .xls ")
    
    colors = []
    if table.index.str.contains('ouleur|olor').any():
        indice = np.where(
            table.index.str.contains('ouleur|olor'))[0][0]
        color_name = table.index[indice]

        # # Creation of dictionnary containing the colors

        # for coul in range(table.iloc[indice].shape[0]):
        #     if isinstance(table.iloc[indice][coul], str):
        #         colors[coul] = table.iloc[indice][coul]
        table.loc[color_name] = table.loc[color_name].replace({np.nan: None})

        colors = table.loc[color_name].to_list()
        
        table = table.drop(labels=color_name)
        
    

    return Mask(table,colors,filename)
