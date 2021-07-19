import unittest

import numpy as np
import numpy.testing as npt
from scipy.sparse.csr import csr_matrix

from pylogit.scipy_utils import identity_matrix



def sparse_assert_equal(a1, a2):
    """Assert equality of two sparse matrices"""
    assert type(a1) is type(a2)
    npt.assert_array_equal(a1.data, a2.data)
    npt.assert_array_equal(a1.indices, a2.indices)
    npt.assert_array_equal(a1.indptr, a2.indptr)


def identity_matrix_assert_equal(a1, a2):
    """Assert equality of two identity matrices"""
    assert type(a1) is type(a2)
    assert a1.n == a2.n


class IdentityMatrixConstructorTests(unittest.TestCase):
    """
    Contains the tests for the `identity_matrix` constructor
    """

    def setUp(self):
        """Create the input data needed to test the `identity_matrix`"""
        
        # Set number rows/columns of an identity matrix
        self.n = 3

        # Set negative number of rows which will throw an error
        self.neg_n = -3

        # Set shape for an identity matrix
        self.shape = (3, 3)

        # Set non-square shape which will lead to a ValueError
        self.non_square_shape = (3, 4)

        # Set wrong shape not consisting of ints
        self.string_shape = ('hello', 'world')

        # Set wrong shape not consisting of ints but whose elements are the same
        # If the elements are the same and the type is not checked, the code
        # will act as if it's a square shape because the elements are the same.
        self.string_square_shape = ('hello', 'hello')

        # Set 1-element tuple which will raise a ValueError because the tuple has to contain two elements
        self.tuple1 = (1,)

        # Set 3-element tuple which will raise a ValueError because the tuple has to contain two elements
        self.tuple3 = (1, 2, 3)

        # Set identity_matrix which can be used to create another identity matrix with the same shape
        self.I = identity_matrix(2)

        # Set arbitrary object type not supported by `identity_matrix`
        self.wrong_arg1 = object()
    
    def test_construct_int(self):
        """Test that constructor works for a valid `int` input"""
        I = identity_matrix(self.n)
        identity_matrix_assert_equal(I, identity_matrix(self.n))

    def test_construct_zero(self):
        """Test that constructor works for `0` input (this is a trivial 0-by-0 matrix)"""
        I = identity_matrix(self.n)
        identity_matrix_assert_equal(I, identity_matrix(self.n))

    def test_construct_neg_int_raises(self):
        """Test that constructor throws ValueError upon a negative `int` input"""
        with self.assertRaises(ValueError):
            I = identity_matrix(self.neg_n)
 
    def test_construct_shape(self):
        """Test that constructor works for a valid square `shape` input"""
        I = identity_matrix(self.shape)
        identity_matrix_assert_equal(I, identity_matrix(self.shape[0]))
    
    def test_construct_non_square_shape_raises(self):
        """Test that constructor throws a ValueError upon a non-square `shape` input"""
        with self.assertRaises(ValueError):
            I = identity_matrix(self.non_square_shape)

    def test_construct_non_int_shape(self):
        """Test that constructor throws a ValueError upon non-int elements in a `shape` input"""
        with self.assertRaises(ValueError):
            I = identity_matrix(self.string_shape)

    def test_construct_non_int_square_shape(self):
        """Test that constructor throws a ValueError upon non-int elements in a `shape` input,
        also when the `shape` is 'square' because the two elements are equal"""
        with self.assertRaises(ValueError):
            I = identity_matrix(self.string_square_shape)

    def test_construct_1_tuple(self):
        """Test that constructor throws a ValueError when a 1-tuple is entered rather than a 2-tuple"""
        with self.assertRaises(ValueError):
            I = identity_matrix(self.tuple1)

    def test_construct_3_tuple(self):
        """Test that constructor throws a ValueError when a 3-tuple is entered rather than a 2-tuple"""
        with self.assertRaises(ValueError):
            I = identity_matrix(self.tuple3)

    def test_construct_identity_matrix(self):
        """Test that constructor works when another `identity_matrix` is entered"""
        I = identity_matrix(self.I)
        identity_matrix_assert_equal(I, self.I)
    
    def test_construct_wrong_type(self):
        """Test that constructor throws a TypeError when a wrong type is entered"""
        with self.assertRaises(TypeError):
            I = identity_matrix(self.wrong_arg1)


class IdentityMatrixTests(unittest.TestCase):
    """
    Contains the tests of the `identity_matrix` class
    """

    def setUp(self):
        """Create the input data needed to test the `identity_matrix`"""

        # Set number number of rows of the identity matrix
        self.n = 3

        # Create identity matrix under test
        self.I = identity_matrix(self.n)

        # Create vector to multiply with `I`
        self.v = np.arange(self.n)

        # Create vector to multiply with `I` which will create an error because
        # it contains too many elements
        self.v_too_long = np.arange(4)

        # Create matrix to multiply with `I`
        self.A = np.arange(12).reshape((self.n, 4))

        # Create sparse matrix to multiply with `I`
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 3])
        data = np.array([1, 2, 3, 4, 5, 6])
        self.C = csr_matrix((data, (row, col)), shape=(self.n, 4))
        
    def test_shape(self):
        """Test that the shape of the identity matrix is as expected."""
        self.assertEqual(self.I.shape, (self.n, self.n))

    def test_nnz(self):
        """Test that the number of non-zero elements is as expected"""
        self.assertEqual(self.I.getnnz(), self.n)
        self.assertEqual(self.I.nnz, self.n)

    def test_mul_vector(self):
        """Test that matrix-vector multiplication `I * v` returns `v` again"""
        npt.assert_array_equal(self.I * self.v, self.v)

    def test_mul_vector_raises(self):
        """Test that matrix-vector multiplication `I * v` with wrong-sized `v` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.I * self.v_too_long

    def test_dot_vector(self):
        """Test that matrix-vector multiplication `I.dot(v)` returns `v` again"""
        npt.assert_array_equal(self.I.dot(self.v), self.v)

    def test_dot_vector_raises(self):
        """Test that matrix-vector multiplication `I.dot(v)` with wrong-sized `v`
        raises an error"""
        with self.assertRaises(ValueError):
            self.I.dot(self.v_too_long)

    def test_matmul_vector(self):
        """Test that matrix-vector multiplication `I @ v` returns `v` again"""
        npt.assert_array_equal(self.I @ self.v, self.v)

    def test_matmul_vector_raises(self):
        """Test that matrix-vector multiplication `I @ v` with wrong-sized `v` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.I @ self.v_too_long

    def test_rmul_vector(self):
        """Test that matrix-vector multiplication `v * I` returns `v` again"""
        npt.assert_array_equal(self.v * self.I, self.v)

    def test_rmul_vector_raises(self):
        """Test that matrix-vector multiplication `v * I` with wrong-sized `v` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.v_too_long * self.I

    def test_rmatmul_vector(self):
        """Test that matrix-vector multiplication `v @ I` returns `v` again"""
        npt.assert_array_equal(self.v @ self.I, self.v)

    def test_rmatmul_vector_raises(self):
        """Test that matrix-vector multiplication `v @ I` with wrong-sized `v` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.v_too_long @ self.I

    def test_mul_multivector(self):
        """Test that matrix multiplication `I * A` returns `A` again"""
        npt.assert_array_equal(self.I * self.A, self.A)

    def test_mul_multivector_raises(self):
        """Test that matrix multiplication `I * A.T` with wrong-sized `A.T` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.I * self.A.T

    def test_dot_multivector(self):
        """Test that matrix multiplication `I * A` returns `A` again"""
        npt.assert_array_equal(self.I.dot(self.A), self.A)

    def test_dot_multivector_raises(self):
        """Test that matrix multiplication `I * A.T` with wrong-sized `A.T` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.I.dot(self.A.T)

    def test_matmul_multivector(self):
        """Test that matrix multiplication `I @ A` returns `A` again"""
        npt.assert_array_equal(self.I @ self.A, self.A)

    def test_matmul_multivector_raises(self):
        """Test that matrix multiplication `I @ A.T` with wrong-sized `A.T` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.I @ self.A.T

    def test_rmul_multivector(self):
        """Test that matrix multiplication `A.T * I` returns `A.T` again"""
        At = self.A.T
        npt.assert_array_equal(At * self.I, At)

    def test_rmul_multivector_raises(self):
        """Test that matrix multiplication `A * I` with wrong-sized `A` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.A * self.I

    def test_rmatmul_multivector(self):
        """Test that matrix multiplication `A.T @ I` returns `A.T` again"""
        At = self.A.T
        npt.assert_array_equal(At @ self.I, At)

    def test_rmatmul_multivector_raises(self):
        """Test that matrix multiplication `A @ I` with wrong-sized `A` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.A @ self.I

    def test_mul_sparse_matrix(self):
        """Test that sparse matrix multiplication `I * C` returns `C` again"""
        sparse_assert_equal(self.I * self.C, self.C)

    def test_mul_sparse_matrix_raises(self):
        """Test that sparse matrix multiplication `I * C.T` with wrong-sized `C.T` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.I * self.C.T

    def test_dot_sparse_matrix(self):
        """Test that sparse matrix multiplication `I.dot(C)` returns `C` again"""
        sparse_assert_equal(self.I.dot(self.C), self.C)

    def test_dot_sparse_matrix_raises(self):
        """Test that sparse matrix multiplication `I.dot(C.T)` with wrong-sized `C.T` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.I.dot(self.C.T)

    def test_matmul_sparse_matrix(self):
        """Test that sparse matrix multiplication `I @ C` returns `C` again"""
        sparse_assert_equal(self.I @ self.C, self.C)

    def test_matmul_sparse_matrix_raises(self):
        """Test that sparse matrix multiplication `I @ C.T` with wrong-sized `C.T` 
        raises an error"""
        with self.assertRaises(ValueError):
            self.I @ self.C.T

    # FIXME Pre-multiplying a sparse matrix with `I` currently gives an error.
    # For our use-case `dh_dv.dot(design)` this doesn't matter, because this is 
    # of the form `I * A` with `A` being a numpy array.
    # def test_rmul_sparse_matrix(self):
    #     """Test that sparse matrix multiplication `C.T * I` returns `C.T` again"""
    #     Ct = self.C.T
    #     npt.assert_array_equal(Ct * self.I, Ct)

    # def test_rmul_sparse_matrix_raises(self):
    #     """Test that sparse matrix multiplication `C * I` with wrong-sized `C` 
    #     raises an error"""
    #     with self.assertRaises(ValueError):
    #         self.C * self.I

    # def test_rmatmul_sparse_matrix(self):
    #     """Test that sparse matrix multiplication `C.T @ I` returns `C.T` again"""
    #     Ct = self.C.T
    #     npt.assert_array_equal(Ct @ self.I, Ct)

    # def test_rmatmul_sparse_matrix_raises(self):
    #     """Test that sparse matrix multiplication `C @ I` with wrong-sized `C` 
    #     raises an error"""
    #     with self.assertRaises(ValueError):
    #         self.C @ self.I

    def test_toarray(self):
        """Test that toarray() gives an identity matrix in numpy array format"""
        npt.assert_array_equal(self.I.toarray(), np.identity(3))
    
    def test_tocsr(self):
        """Test that tocsr() returns an identity matrix in CSR format"""
        row = np.arange(self.n)
        col = np.arange(self.n)
        data = np.ones(self.n)
        csr_expected = csr_matrix((data, (row, col)), shape=(self.n, self.n), copy=False)
        
        sparse_assert_equal(self.I.tocsr(), csr_expected)

    def test___pow__(self):
        """Test that __pow__ method always returns the identity regardless of input"""
        identity_matrix_assert_equal(self.I, self.I ** 2)
        identity_matrix_assert_equal(self.I, self.I ** -1)
        identity_matrix_assert_equal(self.I, self.I ** 0)
        identity_matrix_assert_equal(self.I, self.I ** 1)
        identity_matrix_assert_equal(self.I, self.I ** 0.5)

    def test_copy(self):
        """Test that `copy` returns a copy that's not the same object, but equal to the original"""
        Ic = self.I.copy()
        self.assertIsNot(self.I, Ic)
        identity_matrix_assert_equal(self.I, Ic)

    def test_transpose(self):
        """Test that `transpose` simply returns itself again"""
        self.assertIs(self.I.transpose(), self.I)

    def test_transpose_copy(self):
        """Test that `transpose(copy=True)` simply returns a copy of itself"""
        It = self.I.transpose(copy=True)
        self.assertIsNot(self.I, It)
        identity_matrix_assert_equal(self.I, It)

    def test_conj(self):
        """Test that `conj` simply returns itself again"""
        self.assertIs(self.I.conj(copy=False), self.I)

    def test_conj_copy(self):
        """Test that `conj()` simply returns a copy of itself"""
        It = self.I.conj()
        self.assertIsNot(self.I, It)
        identity_matrix_assert_equal(self.I, It)

    def test_conjugate(self):
        """Test that `conj` simply returns itself again"""
        self.assertIs(self.I.conjugate(copy=False), self.I)

    def test_conjugate_copy(self):
        """Test that `conjugate()` simply returns a copy of itself"""
        It = self.I.conjugate()
        self.assertIsNot(self.I, It)
        identity_matrix_assert_equal(self.I, It)

    def test_H(self):
        """Test that `I.H` (Hermitian transpose) simply returns a copy of itself"""
        Ih = self.I.H
        self.assertIsNot(self.I, Ih)
        identity_matrix_assert_equal(self.I, Ih)
