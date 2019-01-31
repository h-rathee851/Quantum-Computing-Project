"""
Testing random stuff
"""

# Test implementation of Grover's algorithm for 3 qubits with a 3 qubit oracle.
from qc_simulator.qc import *
import numpy as np
import math


def grover5():

    # Define oracle for 5 qubits, where a = 10010 is the "magic number"
    not_gate = Not()
    h_gate = Hadamard()
    I = Operator(base=np.eye(2,2))

    # Define 5-controlled-Not gate (the crude way for now)
    c5_not = CUGate(not_gate, 5)

    oracle_gate = (I % not_gate % not_gate % I % not_gate % I) *\
        c5_not * ( I % not_gate % not_gate % I % not_gate % I)

    # Define the inversion about average operator
    # W = H**n % X**n % (cn^-1 Z) * X%%n * H%%

    # Define z and control z_gates
    z = PhaseShift(np.pi)
    control_z = CUGate(z, 4)
    h_n_gate = Hadamard(6)
    not_n_gate = Not(6)

    W = h_n_gate * not_n_gate * (control_z % I) * not_n_gate * h_n_gate
    test = W * ( Hadamard(6) * QuantumRegister(6) )
    test.remove_aux(1/np.sqrt(2))

    G = W * oracle_gate

    # Define the input and ancillary quantum registers
    input_register = Hadamard(5) * QuantumRegister(5)
    aux = h_gate * not_gate * QuantumRegister()
    register = input_register

    # Loop and apply grover operator iteratively
    n = math.ceil( math.sqrt(5) )
    for i in range(n):
        # Add auxilary qubit to register
        register = register * aux

        # Apply grover iteration
        register = oracle_gate * register
        register.remove_aux(1/np.sqrt(2))
        register = register* aux
        register = W * register
        #register.remove_aux(1/np.sqrt(2))

        # Extract input register and reset auxillary qubit (hacky way)
        register.remove_aux(1/np.sqrt(2))

        aux = h_gate * not_gate * QuantumRegister()


    n = register.measure()

    return n



for i in range(10):
    test = grover5()
    print(test)



