"""
Oracle for simple Grover's Algorith implementation
"""

import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt
import sys
# sys.path.append("..")
try:
    from src.sparse_matrix import SparseMatrix
    from src.quantum_register import QuantumRegister
    from src.quantum_operator import Operator
except:
    from sparse_matrix import SparseMatrix
    from quantum_register import QuantumRegister
    from quantum_operator import Operator

class Oracle(Operator):
    """
    The oracle searches the quantum register and multiples states that fulfil
    the problem condition by -1.
    Here this oracle findes states which are multiples of 5.
    """
    def __init__(self, n_qubits: int=1):
        dimension = 2**n_qubits
        base = np.zeros((dimension, dimension))
        for i in range(dimension):
            if (i % 5 == 0):
                base[i][i] = -1
            else:
                base[i][i] = 1

        super(Oracle, self).__init__(n_qubits, base)

    def showit(self):
        print(self)

class GeneralOracle(Operator):
    """
    The oracle searches the quantum register and multiples states that fulfil
    the problem condition by -1
    """
    def __init__(self, n, query, n_qubits: int=3):
        """
        Set the problem to be solved as f(x) , where the oracle returns the
        answer when f(x) = 0
        :param: (int) n: Find solutions coresponding to n.
        :param: (int) query: Function selection. 1 = multiples of n. 2 = powers
                of n.
        :param: (int) n_qubits: Number of qubits in quantum register.
        """
        dimension = 2**n_qubits
        base = np.zeros((dimension,dimension))
        if (query == 1):  # Finds multiples of n.
            self.shape = lambda t: t % n
            for i in range(dimension):
                if (self.shape(i) == 0):
                    base[i][i] = -1
                else:
                    base[i][i] = 1
        elif(query == 2):  # Finds powers of n.
            self.shape = lambda t: t**(1/n)
            for i in range(dimension):
                if (self.shape(i) % 1 == 0 and i != 0):
                    base[i][i] = -1
                else:
                    base[i][i] = 1
        super(GeneralOracle, self).__init__(n_qubits, base)

    def showit(self):
        print(self)
