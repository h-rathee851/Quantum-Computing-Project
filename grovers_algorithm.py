#******************************************************************************#
#                                                                              #
#                        Main function for Grover's                            #
#                            search algorithm.                                 #
#                                                                              #
#******************************************************************************#

import sys
import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt

from src.grover import Grover

from src.sparse_matrix import SparseMatrix
from src.quantum_register import QuantumRegister
from src.quantum_operator import Operator
from src.operators import *

from src.oracle import *
from src.grover_phase import *



def main(args):
    if len(args) != 2:
        print("Add the number of qubits and target state.")
        print("python grovers_algorithm.py n_qubits target_state")
        sys.exit()

    n_qubits = int(args[0])
    target_state = int(args[1])


    p = Grover(n_qubits, target_state)
    p.build_quantum_register()
    p.init_register()
    p.init_reflection_matrix()
    p.gen_oracle()
    itterations = 1000
    for i in range(itterations):
        sys.stdout.write("Simulation progress: %.1f%%\r"
                                % ((100 * i / itterations)))
        sys.stdout.flush()  # Prints progress of simulation.
        result = p.run(20)
    p.plot_results()

if __name__ == '__main__':
    main(sys.argv[1:])
