"""
:file:      functions_dvf.py
:brief:     Image warping functions using a DVF
:author:    Jens Renders
"""

# This file is part of ImWIP.
#
# ImWIP is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# ImWIP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of
# the GNU General Public License along with ImWIP. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import imwip.numba_backend
try:
    import libimwip
    libimwip_available = True
except ImportError:
    libimwip_available = False


def warp(
        image,
        u,
        v,
        w=None,
        out=None,
        degree=3,
        backend=None
    ):
    """
    Warps a 2D or 3D image along a DVF.

    This function is linear in terms of the input image (even if degree 3
    is used for the splines). Therefore it has an adjoint function which is computed
    by :meth:`adjoint_warp`.

    :param image: Input image
    :param u: First component of the DVF describing the warp
    :param v: Second component of the DVF describing the warp
    :param w: Third component of the DVF describing the warp.
        Leave empty for a 2D warp
    :param out: Array to write the output image in.
        If None, an output array will be allocated.
    :param degree: Degree of the splines used for interpolation
    :param backend: Whether to use the cpp or numba backend. If None, ``cpp`` will be used
        if available, else ``numba``

    :type image: :class:`numpy.ndarray`
    :type u: :class:`numpy.ndarray`
    :type v: :class:`numpy.ndarray`
    :type w: :class:`numpy.ndarray`, optional
    :type out: :class:`numpy.ndarray`, optional
    :type degree: 1 or 3, optional
    :type backend: ``cpp`` or ``numba``, optional
    
    :return: The warped image
    :rtype: :class:`numpy.ndarray`
    """

    if backend is None:
        backend = "cpp" if libimwip_available else "numba"

    dim = 2 if w is None else 3
    if backend == "cpp":
        if dim == 2:
            warp_function = libimwip.warp_2D
        else:
            warp_function = libimwip.warp_3D
    elif backend == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.warp_2D
        else:
            warp_function = imwip.numba_backend.warp_3D
    else:
        raise ValueError("backend should be \"cpp\" or \"numba\"")

    if dim == 2:
        return warp_function(
                image,
                u,
                v,
                out,
                degree
            )
    else:
        return warp_function(
                image,
                u,
                v,
                w,
                out,
                degree
            )

def adjoint_warp(
        image,
        u,
        v,
        w=None,
        out=None,
        degree=3,
        backend=None
    ):
    """
    The function :meth:`warp` is a linear function of the input image (even if degree 3
    is used for the splines). Therefore it has an adjoint function which is computed
    by this function, as described by :cite:t:`renders2021adjoint`. See :meth:`warp`
    for the description of parameters and return value.

    """

    if backend is None:
        backend = "cpp" if libimwip_available else "numba"

    dim = 2 if w is None else 3
    if backend == "cpp":
        if dim == 2:
            warp_function = libimwip.adjoint_warp_2D
        else:
            warp_function = libimwip.adjoint_warp_3D
    elif backend == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.adjoint_warp_2D
        else:
            warp_function = imwip.numba_backend.adjoint_warp_3D
    else:
        raise ValueError("backend should be \"cpp\" or \"numba\"")

    if dim == 2:
        return warp_function(
                image,
                u,
                v,
                out,
                degree
            )
    else:
        return warp_function(
                image,
                u,
                v,
                w,
                out,
                degree
            )


def diff_warp(
        image,
        u,
        v,
        w=None,
        diff_x=None,
        diff_y=None,
        diff_z=None,
        backend=None
    ):
    """
    The derivative of :meth:`warp` towards the DVF. This function assumes splines of degree 3,
    to ensure differentiability.

    :param image: Input image
    :param u: First component of the DVF describing the warp
    :param v: Second component of the DVF describing the warp
    :param w: Third component of the DVF describing the warp.
        Leave empty for a 2D warp
    :param diff_x: Array to write the derivative to the first component in.
        If None, an output array will be allocated.
    :param diff_y: Array to write the derivative to the first component in.
        If None, an output array will be allocated.
    :param diff_z: Array to write the derivative to the first component in.
        If None, an output array will be allocated.
    :param backend: Whether to use the cpp or numba backend. If None, ``cpp`` will be used
        if available, else ``numba``

    :type image: :class:`numpy.ndarray`
    :type u: :class:`numpy.ndarray`
    :type v: :class:`numpy.ndarray`
    :type w: :class:`numpy.ndarray`, optional
    :type diff_x: :class:`numpy.ndarray`, optional
    :type diff_y: :class:`numpy.ndarray`, optional
    :type diff_z: :class:`numpy.ndarray`, optional
    :type backend: ``cpp`` or ``numba``, optional
    :return: diff_x, diff_y, diff_z
    :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, 
    """

    if backend is None:
        backend = "cpp" if libimwip_available else "numba"

    dim = 2 if w is None else 3
    if backend == "cpp":
        if dim == 2:
            warp_function = libimwip.diff_warp_2D
        else:
            warp_function = libimwip.diff_warp_3D
    elif backend == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.diff_warp_2D
        else:
            warp_function = imwip.numba_backend.diff_warp_3D
    else:
        raise ValueError("backend should be \"cpp\" or \"numba\"")

    if dim == 2:
        return warp_function(
                image,
                u,
                v,
                diff_x,
                diff_y,
            )
    else:
        return warp_function(
                image,
                u,
                v,
                w,
                diff_x,
                diff_y,
                diff_z,
            )