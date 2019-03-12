import sys
import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt
import os

from src.grover import Grover

from src.sparse_matrix import SparseMatrix
from src.quantum_register import QuantumRegister
from src.quantum_operator import Operator
from src.operators import *

from src.oracle import *
from src.grover_phase import *

def main():
    c = SQUSwap(n_qubits=4)
    # c = PSWAP(n_qubits=4, phi=(np.pi/3))
    example = QuantumRegister(n_qubits=4)
    example.setState([0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1])
    example = c * example
    print(example)
    example.plotRegister()
    print(example.measure())

if __name__ == '__main__':
    main()
