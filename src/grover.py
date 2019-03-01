#******************************************************************************#
#                                                                              #
#                         Source code for Grover's                             #
#                            search algorithm.                                 #
#                                                                              #
#******************************************************************************#


import numpy as np
from numpy.linalg import norm
import cmath
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


class Grover:
    def __init__(self, n_qubits, target_state):
        self.n_qubits = n_qubits
        self.target_state = target_state
        self.qr = None
        self.D = None
        self.oricle = None

    def build_quantum_register(self):
        self.qr = QuantumRegister(self.n_qubits)
        return self.qr

    def print_register(self):
        print(self.qr)

    def init_register(self):
        h = H(self.n_qubits)
        x = X(self.n_qubits)
        h2 = H(self.n_qubits)
        empty_register = QuantumRegister(self.n_qubits)
        self.qr = h * self.qr
        aux = h * x * empty_register
        self.qr = self.qr * aux
        return self.qr

    def init_reflection_matrix(self):
        h = H(self.n_qubits)
        i = I(self.n_qubits)
        x = X(self.n_qubits)
        cnot = CNOT(2*self.n_qubits)
        a = (h % i)
        b = (x % i)
        c = cnot
        d = (x % i)
        e = (h % i)
        self.D = a * b * c * d * e
        return self.D

    def gen_oracle(self):
        self.oricle = Oracle(self.target_state)
        return self.oracle

    def run(self, k):
        # k is the number of tagged states
        runs = round( ((math.pi / 4) / math.sqrt(k)) * 2**(self.n_qubits / 2))
        for i in range(runs):
            self.qr = self.oracle * self.qr
            self.qr = self.D * self.qr
        result = self.qr.measure()
        return result
