#******************************************************************************#
#                                                                              #
#                  Source code for Quantum part of Shor's                      #
#                          factoring algorithm.                                #
#                                                                              #
#******************************************************************************#

# Import
import numpy as np
import math
import random
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


# Defining a method that returns the state of the register after passing through the black box
def UaGate(N,m,t_qubits):
    # args: N: Number to be factored
    #       m: base of the function (random number between 1 and N-1)
    #       t_qubits: Number of qubits in first register. Determines the probability
    #                 with which we get the right answer.

    # Determine l_qubits, qubits in the second register as the number of qubits required to store N
    l_qubits = math.ceil(math.log(N,2))
    tot_qubits = t_qubits+l_qubits
    QR3 = QuantumRegister(tot_qubits)
    QR3.setState([0]*(2**tot_qubits))
    second_reg_vals_  = []
    for i in range(2**t_qubits):
        res = int(np.mod(m**i, N))
        QR1 = QuantumRegister(t_qubits)
        QR2 = QuantumRegister(l_qubits)
        state_1 = np.zeros(2**t_qubits)
        state_2 = np.zeros(2**l_qubits)
        state_1[i] = 1
        state_2[res] = 1
        QR1.setState(state_1)
        QR2.setState(state_2)
        QR3 = QR3 + (QR1*QR2)
        second_reg_vals_ += [res]
    second_reg_vals_ = np.array(second_reg_vals_)
    second_reg_vals_ = np.unique(second_reg_vals_)
    QR3.normalize()
    return QR3,second_reg_vals_

def measure_second_reg(N,m,t_qubits,second_reg_vals_):
    mes_val = random.choice(second_reg_vals_)
    QR1 = QuantumRegister(t_qubits)
    state_1 = np.zeros(2**t_qubits)
    for i in range(2**t_qubits):
        res = int(np.mod(m**i, N))
        if res == mes_val:
            state_1[i] = 1
    QR1.setState(state_1)
    return QR1
