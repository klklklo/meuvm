import functools
import xarray as xr
from importlib_resources import files


@functools.cache
def _read_coeffs(file):
    return xr.open_dataset(files('seuvm._coeffs').joinpath(file))

def get_lin_seuvm():
    return _read_coeffs('_lin_seuvm_coeffs.nc').copy()

def get_poly_seuvm():
    return _read_coeffs('_poly_seuvm_coeffs.nc').copy()