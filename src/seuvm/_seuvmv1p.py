import numpy as np
import pandas as pd
import xarray as xr
import seuvm._misc as _m


class Seuvmv1p:
    '''
    Poly Seuvm model class.
    '''

    def __init__(self):
        self._dataset = _m.get_seuvm_ver1p()
        self.coeffs = np.array(np.vstack([self._dataset['a'],
                                          self._dataset['b'],
                                          self._dataset['c']])).T

    def get_f(self, f107):
        try:
            if isinstance(f107, float) or isinstance(f107, int):
                return np.array([f107**2, f107, 1.0], dtype=np.float64).reshape(1, 3)
            return np.vstack([np.array([x**2, x, 1.0]) for x in f107], dtype=np.float64)
        except TypeError:
            raise TypeError('Only int, float or array-like object types are allowed.')

    def get_spectral_bands(self, f107):
        x = self.get_f(f107)
        res = np.dot(self.coeffs, x.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'f107'), res),
                                     'lband': ('band_number', self._dataset['lband'].values),
                                     'uband': ('band_number', self._dataset['uband'].values)},
                          coords={'f107': x[:, 1],
                                  'band_center': self._dataset['center'].values,
                                  'band_number': np.arange(190)})

    def get_spectra(self, f107):
        return self.get_spectral_bands(f107)

    def predict(self, f107):
        return self.get_spectral_bands(f107)
