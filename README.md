# MARCIA - MAsking spectRosCopIc dAtacube
## Manual classifier for µXRF and EDS/SEM hypercubes
 - Classification is achieved by defining masks that are linear combination of elemental intensities in spectra.
 - Classes can then be extracted and read with hyperspy or PyMca or Esprit


## Install
Just do 
```bash
pip install git+ssh://git@github.com/hameye/marcia.git
``` 

## Use in python
```python
from marcia.mask import Mask
```
