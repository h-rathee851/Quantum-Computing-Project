# Quantum Fourier Transform.

import numpy as np
from numpy.linalg import norm
import cmath
import math
import random
import matplotlib.pyplot as plt
try:
    from src.sparse_matrix import SparseMatrix
    from src.quantum_register import QuantumRegister
    from src.quantum_operator import Operator
    from src.operators import *
except:
    from sparse_matrix import SparseMatrix
    from quantum_register import QuantumRegister
    from quantum_operator import Operator
    from operators import *




def QFT(n_qubits):
    """
        Returns matrix form of Quantum Fourier Transform acting on n qubits

        :param: (int) n_qubits: Number of qubits QFT acts on
    """
    n = n_qubits
    # Start by applying Hadamard to the first qubit and them iterate to apply the phase gates
    QFT = H(1) % I(n - 1)
    for j in range(1, n):
        base = R_phi(2 * math.pi / (2**(j + 1)))
        c_r = CUGate(base, empty_qw= j - 1, reverse= True)
        if j != n - 1:
            c_r = c_r % I(n - j - 1)
        QFT = c_r * QFT
    for i in range(1, n - 1):
        QFT = (I(i) % H(1) % I(n - i - 1)) * QFT
        for j in range(1, n - i):
            base = R_phi(2 * math.pi / (2**(j + 1)))
            c_r = I(i) % CUGate(base, empty_qw= j - 1, reverse = True)
            if j != n - i - 1:
                c_r = c_r % I(n - i - j - 1)
            QFT = c_r * QFT
    # Apply Hadamard on the last qubit
    QFT = (I(n-1) % H(1)) * QFT
    QFT = SWAP(n) * QFT
    return QFT



def invQFT(n_qubits):
    """
    returns matrix form of inverse Quantum Fourier Transform acting on n qubits

    :param: (int) n_qubits: Number of qubits QFT acts on
    """
    # Return the Hermitian Transpose(Inverse) of QFT
    return QFT(n_qubits).getHermTranspose()
