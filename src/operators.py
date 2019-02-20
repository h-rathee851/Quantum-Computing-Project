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
    def __init__(self, n_qubits: int=1):
        # Define "base" hadamard matrix for one qubit and correponding sparse matrix
        base = 1 / np.sqrt(2) * np.array([[1, 1],
                                        [1, -1]])
        super(Hadamard, self).__init__(n_qubits, base)

class CNot(Operator):
    """
    Controlled not gate.
    """
    def __init__(self, n_qubits: int=2):  # Check if n_qubits should be 2.
        base = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
        super(CNot, self).__init__(n_qubits, base)

class XGate(Operator):
    """
    X gate or Bit flip gate.
    """
    def __init__(self, n_qubits: int=1):
        base = np.array([[0, 1],
                        [1, 0]])
        super(XGate, self).__init__(n_qubits, base)

class ZGate(Operator):
    """
    Z gate or phase flip gate.
    """
    def __init__(self, n_qubits: int=1):
        base = np.array([[1, 0],
                       [0,-1]])
        super(ZGate, self).__init__(n_qubits, base)

class TGate(Operator):
    """
    T gate or phase shift gate.
    """
    def __init__(self, n_qubits: int=1, theta):
        base = np.array([[1, 0],
                        [0, np.exp(1j * theta)]])
        super(TGate, self).__init__(n_qubits, base)

class CVGate(Operator):
    def __init__(self, n_qubits: int=2, theta):  # Check if n_qubits should be 2.
        base = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1j]])
        super(CVGate, self).__init__(n_qubits, base)

def test():
    # H_1 = Hadamard(1)
    #
    # H_2 = Hadamard(1)
    #
    # print((H_1%H_2))
    cn = XGate()
    print(cn)

# test()
