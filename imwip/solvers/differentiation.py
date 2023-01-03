"""
:file:      differentiation.py
:brief:     provides differentiation functionality to
            ImWIP operators and pylops operators that contain ImWIP operators.
:author:    Jens Renders
"""


import numpy as np
import pylops

def diff(A, x, to=None):
    """
    Given an imwip operator :math:`A = A(p)` (where :math:`p` represents all the parameters of
    :math:`A`), or a :py:mod:`pylops` block operator built of those, this function gives the derivative
    of :math:`A(p)x` towards :math:`p`.

    :param A: An ImWIP image warping operator or a Pylops block operator containing
        ImWIP image warping operators.
    :param x: A raveled image on which A acts
    :param to: a parameter or list of parameters to which to differentiate.

    :type A: :class:`~scipy.sparse.linalg.LinearOperator`
    :type x: :class:`numpy.ndarray`
    :type to: string or sequence of strings, optional

    :return: The derivative or list of derivatives towards the parameters specified in `to`
    :rtype: :class:`~scipy.sparse.linalg.LinearOperator` or list of
        :class:`~scipy.sparse.linalg.LinearOperator`
    """

    if hasattr(A, "derivative"):
        if isinstance(to, str):
            return A.derivative(x, to=[to])[0]
        else:
            return A.derivative(x, to=to)
    elif isinstance(A, pylops.Identity):
        return [np.zeros((x.size, 0)) for var in to]
    elif isinstance(A, pylops.VStack):
        if to is None:
            return pylops.BlockDiag([diff(Ai, x) for Ai in A.ops])
        elif isinstance(to, str):
            derivatives = []
            for Ai in A.ops:
                derivatives.append(diff(Ai, x, to=[to])[0])
            return pylops.BlockDiag(derivatives)
        else:
            diff_dict = {var: [] for var in to}
            for Ai in A.ops:
                derivatives = diff(Ai, x, to=to)
                for var, derivative in zip(to, derivatives):
                    diff_dict[var].append(derivative)
            return [pylops.BlockDiag(diff_dict[var]) for var in to]
    elif isinstance(A, pylops.BlockDiag):
        if to is None:
            raise NotImplementedError()
        elif isinstance(to, str):
            derivatives = []
            index = 0
            for Ai in A.ops:
                derivatives.append(diff(Ai, x[index:index + Ai.shape[1]], to=[to])[0])
                index += Ai.shape[1]
            return pylops.BlockDiag(derivatives)
        else:
            diff_dict = {var: [] for var in to}
            index = 0
            for Ai in A.ops:
                derivatives = diff(Ai, x[index:index + Ai.shape[1]], to=to)
                index += Ai.shape[1]
                for var, derivative in zip(to, derivatives):
                    diff_dict[var].append(derivative)
            return [pylops.BlockDiag(diff_dict[var]) for var in to]