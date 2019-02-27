#******************************************************************************#
#                                                                              #
#                         Source code for Shor's                               #
#                          factoring algorithm.                                #
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
