import functools
import xarray as xr
from importlib_resources import files


@functools.cache
def _read_coeffs(file):
    return xr.open_dataset(files('yaeuvm._coeffs').joinpath(file))

def get_yaeuvm_ba():
    return _read_coeffs('_yaeuvm_ba_coeffs.nc').copy()

def get_yaeuvm_r():
    return _read_coeffs('_yaeuvm_r.nc').copy()

def get_seuvm_ver1p():
    return _read_coeffs('_seuvmv1p_coeffs.nc').copy()

def get_yaeuvm_br():
    return _read_coeffs('_yaeuvm_br.nc').copy()
