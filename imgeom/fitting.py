# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import numpy as np
from astropy.modeling import models, fitting
from .reproject import reproject


__all__ = ['extract_reprojected_region', 'quadratic_peakfinder']


def extract_reprojected_region(x, y, rectified_wcs, unrectified_data,
                               radius=5):
    """
    Extract pixel coordinates and values from unrectified data based
    on a peak pixel location in the rectified data.

    Parameters
    ----------
    x, y : float
        Pixel location in the rectified frame.

    rectified_wcs : `~astropy.wcs.WCS`
        WCS of the rectified (e.g. drizzled) frame.

    unrectified_data : list of tuples
        A list of 2-element tuples containing the unrectified image
        data.  Each tuple corresponds to one unrectified image and
        contains its ``(data, wcs)``, where ``data`` is the 2D image
        and ``wcs`` is a `~astropy.wcs.WCS` object.

    radius : float, optional
        The radius of the region to extract around the peak pixel in the
        unrectified data.

    Returns
    -------
    x_rect : `~numpy.ndarray`
        An array of the ``x`` pixel coordinates in the rectified frame.

    y_rect : `~numpy.ndarray`
        An array of the ``y`` pixel coordinates in the rectified frame.

    values : `~numpy.ndarray`
        An array of the pixel values taken from the *unrectified* frame.
    """

    x_rect = []
    y_rect = []
    values = []
    for (data, wcs) in unrectified_data:
        # find the peak pixel in the unrectified data
        f = reproject(rectified_wcs, wcs)
        xcen_unrect, ycen_unrect = f([x], [y])
        xcen_unrect, ycen_unrect = xcen_unrect[0], ycen_unrect[0]

        # extract a circular region from the unrectified data
        yy, xx = np.indices(data.shape)
        yy -= ycen_unrect
        xx -= xcen_unrect
        r = np.sqrt(xx**2 + yy**2)
        idx = np.where(r <= radius)
        values.append(data[idx])

        f2 = reproject(wcs, rectified_wcs)
        tmp_x_rect, tmp_y_rect = f2(idx[1], idx[0])
        x_rect.append(tmp_x_rect)
        y_rect.append(tmp_y_rect)

    return np.ravel(x_rect), np.ravel(y_rect), np.ravel(values)


def quadratic_peakfinder(x, y, z):
    """
    Fit for the peak ``(x, y)`` position by fitting the data with a
    2nd-degree 2D polynomial.

    Parameters
    ----------
    x : array-like
        Input coordinates.

    y : array-like
        Input coordinates.

    z : array-like
        Input values.

    Returns
    -------
    x, y : float
        The ``x`` and ``y`` position of the fitted peak value.
    """

    p_init = models.Polynomial2D(degree=2)   # default coefficients are zero
    fitter = fitting.LinearLSQFitter()
    p_fit = fitter(p_init, x, y, z)

    y_peak = ((2*p_fit.c2_0*p_fit.c0_1 - p_fit.c1_0*p_fit.c1_1) /
              (p_fit.c1_1**2 - 4 * p_fit.c0_2 * p_fit.c2_0))
    x_peak = -(p_fit.c0_1 + 2*p_fit.c0_2 * y_peak) / p_fit.c1_1

    return x_peak, y_peak
