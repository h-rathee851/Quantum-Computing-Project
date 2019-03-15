#******************************************************************************#
#                                                                              #
#                        Main function for Shor's                              #
#                          factoring algorithm.                                #
#                                                                              #
#******************************************************************************#

import sys
import random
import math
import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt
from src.Shors_algorithm import *
from src.sparse_matrix import SparseMatrix
from src.quantum_register import QuantumRegister
from src.quantum_operator import Operator
from src.operators import *
from src.QFT import *
from src.quantum_shor import *

def main(args):
    if len(args) != 2:
        print("python shors_algorithm.py n_qubits test_value")
        sys.exit()

    n_qubits = int(args[0])
    n = int(args[1])

    all_Shor(n, n_qubits)


if __name__ == '__main__':
    main(sys.argv[1:])
