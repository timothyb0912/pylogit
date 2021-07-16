# -*- coding: utf-8 -*-
"""
Created on Thu Jul 7 20:05:51 2021

@name:      Scipy Utilities
@author:    Mathijs van der Vlies
@summary:   Contains an efficient implementation of an identity matrix through the
            identity_matrix class, which simply performs the identity operation
            when multiplied with another matrix without multiplying the individual
            elements.
"""
from __future__ import absolute_import
from scipy.sparse import spmatrix
from scipy.sparse.csr import csr_matrix
import numpy as np
from scipy.sparse.sputils import check_shape, isshape    


class identity_matrix(spmatrix):
    """Efficient implementation of an identity matrix.
    
    When multiplied with another matrix `A`, simply passes through and returns `A`
    without multiplying the individual elements.

    Parameters
    ----------
    arg1 : int, 2-tuple or identity_matrix
        If int `n`, then the matrix is n-by-n.
        If a 2-tuple, then it must be of the form `(n, n)` (square matrix), so that 
        the matrix becomes n-by-n.
        If of type `identity_matrix`, then its size will be copied over.
    copy : bool, default = False
        Inactive parameter because no data is stored by the object, kept for 
        consistency with `spmatrix` interface.
    """

    def __init__(self, arg1, copy=False):
        super().__init__()
        if isinstance(arg1, int):
            self.n = arg1
        elif isinstance(arg1, type(self)):
            self.n = arg1.n
        elif isinstance(arg1, tuple):
            if isshape(arg1):
                m, n = check_shape(arg1)
                if m != n:
                    raise ValueError("Only a square shape is valid.")
                self.n = n
            else:
                raise ValueError("Only a square shape is valid.")
        else:
            raise TypeError(f"Invalid input to constructor. Expected an object of "
                            f"type `int` or {type(self)}")
    
    def set_shape(self, shape):
        return super().set_shape(shape)
    
    def get_shape(self):
        return (self.n, self.n)
    
    shape = property(fget=get_shape, fset=set_shape)

    def getnnz(self, axis=None):
        if axis is None:
            return self.n
        
        return 1

    def __repr__(self):
        return f"<{self.n}x{self.n} identity matrix>"

    def _mul_vector(self, other):
        return other
    
    def _mul_multivector(self, other):
        return other
    
    def _mul_sparse_matrix(self, other):
        return other

    def transpose(self, axes=None, copy=False):
        """
        Reverses the dimensions of the sparse matrix.

        Parameters
        ----------
        axes : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value.
        copy : bool, optional
            Indicates whether or not attributes of `self` should be
            copied whenever possible. The degree to which attributes
            are copied varies depending on the type of sparse matrix
            being used.

        Returns
        -------
        p : `self` with the dimensions reversed.

        See Also
        --------
        numpy.matrix.transpose : NumPy's implementation of 'transpose'
                                 for matrices
        """
        if copy:
            return self.copy()
        
        return self
    
    def conj(self, copy=True):
        """Element-wise complex conjugation.

        If the matrix is of non-complex data type and `copy` is False,
        this method does nothing and the data is not copied.

        Parameters
        ----------
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : The element-wise complex conjugate.

        """
        return self.transpose(copy=copy)    

    def tocsr(self, copy=False):
        """Convert this matrix to Compressed Sparse Row format.

        For this identity matrix, `copy` doesn't affect anything 
        because it doesn't store any underlying data.
        """
        row = np.arange(self.n)
        col = np.arange(self.n)
        data = np.ones(self.n)

        return csr_matrix((data, (row, col)), shape=self.shape, copy=False)

    def __pow__(self, other):
        return self.copy()
