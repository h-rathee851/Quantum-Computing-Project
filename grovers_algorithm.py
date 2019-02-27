#******************************************************************************#
#                                                                              #
#                        Main function for Grover's                            #
#                            search algorithm.                                 #
#                                                                              #
#******************************************************************************#


import matplotlib.pyplot as plt
import numpy as np
from src.quantum_register import QuantumRegister
from src.quantum_operator import Operator
from src.sparse_matrix import SparseMatrix
from src.operators import H, I


a = QuantumRegister(2)
i = I()
print(i)
print(a)
