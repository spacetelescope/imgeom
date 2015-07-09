from __future__ import division, print_function, unicode_literals, absolute_import

# THIRD-PARTY
import numpy as np

# LOCAL
from drizzle import util, calc_pixmap, cdrizzle
from astropy.io import fits

"""
A module designed to facilitate drizzling masks.
"""
__version__ = "0.1.0"
__author__ = "Mihai Cara"
__date__ = "9-July-2015"

_supported_comparisons = ['==', '!=', '<>', '>', '>=', '<', '<=',
                          'eq', 'ne', 'gt', 'ge', 'lt', 'le']

def drizzlemask(inmask, coord_map, in_origin=(1,1), inwht=None,
                compval=0.5, comp='>',
                out_shape=None, out_origin=(1,1), scale=1.0,
                pixfrac=1.0, kernel='square', dtype=None,
                verbose=True):
    """
    Resample a mask using "drizzle" algorithm and then apply thresholding to
    resampled data to create "drizzled" mask.

    Parameters
    ----------

    inmask : numpy.ndarray
        A 2D input mask to be resampled. Before being resampled, non-zero
        elements in the mask are replaced with 1. This is peformed on a copy
        of the `inmask` such that `inmask` itself is not modified.

    coord_map : callable func(x, y)
        A function that converts coordinates from the input mask's coordinate
        frame to the desired output coordinate frame. Must be able to accept
        1D vectors of coordinates and return a tuple of two 1D transformed
        vectors.

    in_origin : int, float, tuple, or list, optional
        Spacial coordinates of the first pixel (i.e., `inmask[0,0]`) in the
        input mask `inmask`. If a `list` or `tuple` is provided, then it must
        have exactly two elements: (x_origin, y_origin). If a single value is
        provided then both coordinates are set equal to the provided value.

    inwht : numpy.ndarray, optional
        A 2D numpy array containing the weights of pixels in the input mask.
        Must have the same dimenstions as `inmask`. If none is supplied,
        the weghting is set to one for all pixels.

    compval : float, optional
        The meaning of this parameter depends on the value of the `comp`
        parameter. When `comp='>'`, `compval` has a meaning of a threshold and
        indicates that values in the drizzled mask larger than `compval` should
        be converted 1 in the binary output mask and values less or equal
        to `compval` should be converted to 0.

    comp : str, optional
        Comparison operation to be performed on drizzled mask deciding if
        a pixel in the output mask will be assigned 1 or 0. When the result
        of comparison between the pixel value in the drizzled mask
        and the value specified by the `compval` parameter is true,
        the corresponding pixel in the output mask is assigned value 1 and
        0 otherwise.

    out_shape : tuple, list, optional
        Specifies the shape of the output mask. This parameter must follow the
        same convention as used in `numpy` for constructing arrays, that is,
        it must be in the form (nrows,ncol), If set to `None`, the shape is
        automatically determined so that all "True" (or 1) pixels in the input
        mask fit in the output mask and the `out_origin` parameter is ignored.

    out_origin : tuple, list, optional
        Spacial coordinates of the first pixel (i.e., with indices `[0,0]`)
        in the output mask. It must have exactly two elements:
        (x_origin, y_origin). This parameter is ignored when `out_shape`=`None`.

    scale : float, optional
        The ratio of the output pixel scale to the input pixel scale.

    pixfrac : float, optional
        The fraction of a pixel that the pixel flux is confined to. The
        default value of 1 has the pixel flux evenly spread across the image.
        A value of 0.5 confines it to half a pixel in the linear dimension,
        so the flux is confined to a quarter of the pixel area when the square
        kernel is used.

    kernel: str, optional
        The name of the kernel used to combine the input. The choice of
        kernel controls the distribution of flux over the kernel. The kernel
        names are: "square", "gaussian", "point", "tophat", "turbo", "lanczos2",
        and "lanczos3". The square kernel is the default.

    dtype : numpy data-type, optional
        Specifies the data type of the output mask. By default, the data-type
        is inferred from the input data: when `inmask` is an `numpy` array of
        integer or boolean data type, the output mask will be of the same type
        as `inmask` and it will be `numpy.int8` otherwise.

    verbose : bool, optional
        Specifies whether to print information and warning messages.


    Returns
    -------
    outmask : numpy.ndarray
        Output binary (1 or 0) mask obtained by drizzling the input mask and
        applying some threshold to the drizzled results. When `out_shape` is
        `None` and the input mask is all 0, then the output mask will be
        an empty array (with no elements).

    out_origin : tuple
        The coordinate of the first pixel in the `outmask` (i.e.,
        `outmask[0,0]`). When `out_shape` is not `None` the value of
        `out_origin` is passed directly from input. When `out_shape` is
        `None` and the input mask is all 0, then `out_origin` will be
        set to `(None, None)`.

    """
    # verify that comparison operator is of supported type:
    comp = (''.join(comp.split())).lower()
    if comp not in _supported_comparisons:
        raise ValueError("Unsupported comparison operator")

    # set fillval:
    fillval = ''

    # set stepsize which indicates how often points at the perimeter of the
    # input mask should be evaluated for the purpose of computing the output
    # mask's shape:
    stepsize = 1

    # shape of the input mask:
    inmask = np.asarray(inmask)
    ny, nx = inmask.shape

    # check that input weights (if provided) have the same shape as input mask:
    if inwht is not None:
        inwht = np.asarray(inwht, dtype=np.float32)
        if inwht.shape != inmask.shape:
            raise ValueError("'inwht' array must have the same shape as the "
                             "input mask array")

    # make sure origins are iterable. If not - convert to tuples:
    if not hasattr(in_origin, '__iter__'):
        in_origin = (in_origin, in_origin)
    else:
        if len(in_origin) != 2:
            raise ValueError("'in_origin' must be a either a list "
                             " or a tuple with two elements (coordinates)")

    if out_origin is not None:
        if not hasattr(out_origin, '__iter__'):
            out_origin = (out_origin, out_origin)
        else:
            if len(out_origin) != 2:
                raise ValueError("'out_origin' must be a either a list "
                                 " or a tuple with two elements (coordinates)")

    # data type of the input mask:
    int_inmask = np.issubdtype(inmask.dtype, np.bool_) or \
        np.issubdtype(inmask.dtype, np.integer)

    # data type of the output result:
    if dtype is None:
        if int_inmask:
            odtype = inmask.dtype
        else:
            odtype = np.int8
    else:
        odtype = dtype

    # auto determine output image size?
    auto_out_shape = out_shape is None

    ###############################################
    ##  To save time on drizzling, find out      ##
    ##  how much of the input mask needs to be   ##
    ##  drizzled and how big the output may be.  ##
    ###############################################

    #1. find the bounding box of the non-zero pixels in the input mask:
    xmin, xmax, mbnx, ymin, ymax, mbny = _min_box(inmask, int_inmask, True)
    # if there are no non-zero pixels return an empty array (if out_shape==None)
    # or an array of zeros (if out_shape!=None):
    if mbnx*mbny == 0:
        if auto_out_shape:
            if verbose:
                print("WARNING: No non-zero data have been found in the input "
                      "mask.\nWARNING: Output mask will be empty.")
            return (np.empty((mbny, mbnx), dtype=odtype), (None, None))
        else:
            return (np.zeros((mbny, mbnx), dtype=odtype), out_origin)

    #2. Expand the bounding box by 'ceil(scale)' pixel:
    pad = int(np.ceil(scale))
    xmin = max(0, xmin-pad)
    xmax = min(nx-1, xmax+pad)
    ymin = max(0, ymin-pad)
    ymax = min(ny-1, ymax+pad)
    minview = np.s_[ymin:ymax+1,xmin:xmax+1]

    #3. create smallest-sized numpy.float32 copy of input mask with
    #   element values 0 or 1:
    in_mask_min = np.zeros((ymax-ymin+1, xmax-xmin+1), dtype=np.float32)
    non_zeros = (inmask != 0)[minview]
    in_mask_min[non_zeros] = 1.0

    #4. find bounding box for output image:

    # - compute the coordinates of the border
    nintervals = np.ceil((xmax-xmin+1) / stepsize)
    xr = np.linspace(xmin-0.5,xmax+0.5,nintervals,dtype=np.float)
    nxr = xr.shape[0]
    yr = np.linspace(ymin-0.5,ymax+0.5,nintervals,dtype=np.float)[1:-1]
    nyr = yr.shape[0]
    npts = 2 * (nxr+nyr)
    in_borderx = np.empty((npts,), dtype=np.float)
    in_bordery = np.empty((npts,), dtype=np.float)
    in_borderx[0:nxr] = xr
    in_bordery[0:nxr] = ymin-0.5
    sl = np.s_[nxr:2*nxr]
    in_borderx[sl] = xr
    in_bordery[sl] = ymax+0.5
    sl = np.s_[2*nxr:2*nxr+nyr]
    in_borderx[sl] = xmin-0.5
    in_bordery[sl] = yr
    sl = np.s_[2*nxr+nyr:]
    in_borderx[sl] = xmax+0.5
    in_bordery[sl] = yr
    in_borderx += in_origin[0]
    in_bordery += in_origin[1]

    # - convert these coordinates to the output frame:
    out_borderx, out_bordery = coord_map(in_borderx, in_bordery)
    oxmin = np.amin(out_borderx)
    oxmax = np.amax(out_borderx)
    oymin = np.amin(out_bordery)
    oymax = np.amax(out_bordery)

    onx = int(np.ceil(oxmax-oxmin+1))
    ony = int(np.ceil(oymax-oymin+1))

    #5. if the shape of the output mask was not provided, use the minimal
    #   bounding box for output mask to define output shape:
    if out_shape is None:
        out_shape = (ony-1, onx-1)
        out_origin = (oxmin+0.5, oymin+0.5)
    else:
        if not _do_rect_intersect(
            oxmin, oxmax, oymin, oymax,
            out_origin[0]-0.5, out_origin[0]+out_shape[1]+0.5,
            out_origin[1]-0.5, out_origin[1]+out_shape[0]+0.5
            ):
            return (np.zeros(out_shape, dtype=odtype), out_origin)

    #6. create pixel map:

    # - create a list of pixel coordinates in the minimal mask:
    in_y, in_x = np.indices(in_mask_min.shape, dtype=np.float)
    in_x += xmin + in_origin[0]
    in_y += ymin + in_origin[1]

    # - compute pixel coordinates in the output (undistorted) frame:
    out_x, out_y = coord_map(in_x, in_y)

    # - create pixel map array:
    pixmap = np.dstack((out_x, out_y))

    #7. adjust pixmap coordinates to output origin:
    pixmap -= np.asarray(out_origin, dtype=np.float)

    ##########################
    ##  Drizzle the mask:   ##
    ##########################

    #1. create array for output drizzled mask and dummy arrays for output weight
    #   and context:
    drizmask = np.zeros(out_shape, dtype=np.float32)
    outwht = np.zeros(out_shape, dtype=np.float32)
    outctx = np.zeros(out_shape, dtype=np.int32)

    #2. create an array of weights if not provided:
    if inwht is None:
        in_wht_min = np.ones_like(in_mask_min)
    else:
        in_wht_min = inwht[minview]

    #3. drizzle:
    _vers, nmiss, nskip = cdrizzle.tdriz(
        input=in_mask_min, weights=in_wht_min, pixmap=pixmap,
        output=drizmask, counts=outwht, context=outctx,
        uniqid=1, xmin=0, xmax=0,
        ymin=0, ymax=0, scale=scale, pixfrac=pixfrac,
        kernel=kernel, in_units='cps', expscale=1.0,
        wtscale=1.0, fillstr=fillval)

    #4. apply threshold to the output mask:
    if comp == '>' or comp == 'gt':
        valid_pix = np.greater(drizmask, compval)
    elif comp == '>=' or comp == 'ge':
        valid_pix = np.greater_equal(drizmask, compval)
    elif comp == '<' or comp == 'lt':
        valid_pix = np.less(drizmask, compval)
    elif comp == '<=' or comp == 'le':
        valid_pix = np.less_equal(drizmask, compval)
    elif comp == '==' or comp == 'eq':
        valid_pix = np.isclose(drizmask, compval)
    elif comp == '!=' or comp == '<>' or comp == 'ne':
        valid_pix = np.logical_not(np.isclose(drizmask, compval))
    else:
        raise ValueError("Unsupported comparison operator")

    #5. create output mask while trimming it to minimal boundng box:
    if auto_out_shape:
        oxminf, oxmaxf, ombnx, oyminf, oymaxf, ombny = _min_box(
            valid_pix.astype(dtype=np.int8),
            True, False
        )
        if ombnx*ombny == 0:
            if auto_out_shape:
                if verbose:
                    print("WARNING: No non-zero data have been found in the "
                          "output drizzled mask.\nWARNING: "
                          "Output mask will be empty.")
                return (np.empty((ombny, ombnx), dtype=odtype), (None,None))
            else:
                return (np.zeros((ombny, ombnx), dtype=odtype), out_shape)

        outmask = np.zeros((oymaxf-oyminf+1, oxmaxf-oxminf+1), dtype=odtype)
        outmask[valid_pix[oyminf:oymaxf+1, oxminf:oxmaxf+1]] = 1

    else:
        outmask = np.zeros_like(drizmask, dtype=odtype)
        outmask[valid_pix] = 1

    return (outmask, out_origin)


def _min_box(image, int_type, make_abs=True):
    """
    Return the coordinates of the minimal bounding box that encloses
    all pixels != 0.
    """
    sum_dtype = np.int if int_type else np.float64

    aimage = np.abs(image) if make_abs else image

    margx = np.sum(aimage, axis=0, dtype=sum_dtype)
    nzx = np.nonzero(margx)[0]
    if nzx.shape[0] > 0:
        xmin = nzx[0]
        xmax = nzx[-1]
        nx = xmax-xmin+1
    else:
        xmin = None
        xmax = None
        nx = 0

    margy = np.sum(aimage, axis=1, dtype=sum_dtype)
    nzy = np.nonzero(margy)[0]
    if nzy.shape[0] > 0:
        ymin = nzy[0]
        ymax = nzy[-1]
        ny = ymax-ymin+1
    else:
        ymin = None
        ymax = None
        ny = 0

    return (xmin, xmax, nx, ymin, ymax, ny)


def _do_rect_intersect(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2):
    """
    Return True if two rectangles intersect.
    """
    return (xmax2 > xmin1 and xmax1 > xmin2 and
            ymax2 > ymin1 and ymax1 > ymin2)
