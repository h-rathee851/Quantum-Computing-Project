# Grover phase operator. Kept seperate from other operators as is specific to
# Grover's algorithm.

import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt
try:
    from src.sparse_matrix import SparseMatrix
    from src.quantum_register import QuantumRegister
    from src.quantum_operator import Operator
except:
    from sparse_matrix import SparseMatrix
    from quantum_register import QuantumRegister
    from quantum_operator import Operator

class G_Phase(Operator):
    """
    Grover phase operator, sepcifit to Grover's algorithm for identifying target
    states from a set of states.
    """
    def __init__(self, n_qubits: int=1):
        dimension = 2**n_qubits
        base = np.zeros((dimension,dimension))
        for i in range(dimension):
            if (i  == 0):
                base[i][i] = 1
            else:
                base[i][i] = -1
        super(G_Phase, self).__init__(n_qubits, base)

    def showit(self):
        print(self)
