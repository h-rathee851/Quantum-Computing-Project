#******************************************************************************#
#                                                                              #
#                        Main function for Shor's                              #
#                          factoring algorithm.                                #
#                                                                              #
#******************************************************************************#


import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt

from src.shor import *

from src.sparse_matrix import SparseMatrix
from src.quantum_register import QuantumRegister
from src.quantum_operator import Operator
from src.operators import *


def main():
    a = QuantumRegister(2)
    i = I()
    print(i)
    print(a)


if __name__ == '__main__':
    main()
