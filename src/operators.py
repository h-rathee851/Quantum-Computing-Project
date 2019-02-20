import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt
from sparse_matrix import SparseMatrix
from quantum_operator import Operator  # Couldn't just be called operator as operator already exists.


class Hadamard(Operator):
    """
    Class that defines hadamard gate. This class extends the Operator class.
    """
    def __init__(self, n_qubits: int =1):
        # Define "base" hadamard matrix for one qubit and correponding sparse matrix
        base = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        super(Hadamard, self).__init__(n_qubits, base)


def test():
    H_1 = Hadamard(1)

    H_2 = Hadamard(1)

    print((H_1%H_2))

test()
