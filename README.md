
# MARCIA - MAsking spectRosCopIc dAtacube
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3929745.svg)](https://doi.org/10.5281/zenodo.3929745)
[![PyPI](https://img.shields.io/badge/MARCIA-v0.1.3-blue.svg?maxAge=2592000)](https://pypi.org/project/MARCIA/)

## Manual classifier for µXRF and EDS/SEM hypercubes
 - Classification is achieved by defining masks that are linear combination of elemental intensities in spectra.
 - Classes can then be extracted and read with hyperspy or PyMca or Esprit


## Install
Just do
```bash
pip install marcia
```

## Use in python
```python
from marcia.mask import Mask
```

## Gallery
![Example](https://github.com/hameye/MARCIA/blob/master/gallery.png)
