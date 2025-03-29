import functools
import xarray as xr
from importlib_resources import files


@functools.cache
def _read_coeffs(file):
    return xr.open_dataset(files('seuvm._coeffs').joinpath(file))

def get_seuvm():
    return _read_coeffs('_seuvm_coeffs.nc').copy()