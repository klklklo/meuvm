import numpy as np
import xarray as xr
import yaeuvm._misc as _m


class YaeuvmBa:
    '''
    YAEUVM Binned Average model class.
    '''
    def __init__(self):
        self._dataset = _m.get_yaeuvm_ba()
        self._coeffs = np.array(self._dataset[[f'{i+0.5}irr' for i in range(190)]].to_dataarray())

    def _get_coeffs(self, _f107):
        spectra = np.empty((190, 0))
        for f107 in _f107:
            if f107 <= 80:
                i = 0
            elif 80 < f107 <= 100:
                i = 1
            elif 100 < f107 <= 120:
                i = 2
            elif 120 < f107 < 140:
                i = 3
            elif 140 < f107 < 160:
                i = 4
            elif 160 < f107 < 180:
                i = 5
            elif 180 < f107 < 200:
                i = 6
            elif 200 < f107 < 220:
                i = 7
            elif 220 < f107 < 240:
                i = 8
            elif 240 < f107 < 260:
                i = 9
            elif f107 > 260:
                i = 10

            spectrum = self._coeffs[:, i].reshape((190, 1))
            spectra = np.hstack([spectra, spectrum])

        return spectra

    def get_spectral_bands(self, f107):
        f107 = np.array([f107], dtype=np.float64) if isinstance(f107, (int, float)) \
            else np.array(f107, dtype=np.float64)

        res = self._get_coeffs(f107)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'f107'), res),
                                     'lband': ('band_number', np.arange(0,190)),
                                     'uband': ('band_number', np.arange(1,191))},
                          coords={'f107': f107,
                                  'band_center': [i+0.5 for i in range(190)],
                                  'band_number': np.arange(190)})

    def get_spectra(self, f107):
        f107 = np.array([f107], dtype=np.float64) if isinstance(f107, (int, float)) \
            else np.array(f107, dtype=np.float64)

        return self.get_spectral_bands(f107)

    def predict(self, f107):
        f107 = np.array([f107], dtype=np.float64) if isinstance(f107, (int, float)) \
            else np.array(f107, dtype=np.float64)

        return self.get_spectral_bands(f107)
