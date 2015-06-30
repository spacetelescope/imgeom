from astropy.io import fits
from astropy import wcs as fitswcs
from gwcs import wcs
from gwcs import utils as gwutil
from ..reproject import reproject
from astropy.utils.data import get_pkg_data_filename

from numpy.testing import utils

def test_reproject():
    hdr1 = fits.Header.fromfile(get_pkg_data_filename('data/simple_wcs.hdr'))
    hdr2 = fits.Header.fromfile(get_pkg_data_filename('data/simple_wcs2.hdr'))
    w1 = fitswcs.WCS(hdr1)
    w2 = fitswcs.WCS(hdr2)
    gw2 = wcs.WCS(output_frame='icrs',
                  forward_transform=gwutil.make_fitswcs_transform(hdr2))
    func1 = reproject(w1, w2)
    func2 = reproject(w1, gw2)
    x1, y1 = func1(1, 2)
    x2, y2 = func2(1, 2)
    utils.assert_allclose(x1, x2, atol=10**-7)
    utils.assert_allclose(y1, y2, atol=10**-7)
