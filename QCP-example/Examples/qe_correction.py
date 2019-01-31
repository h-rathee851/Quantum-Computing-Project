"""
University of Edinburgh: School of Physics and Astronomy
Quantum Computing Project

Functions implementing quantum error correction.
"""

from qc_simulator.functions import *
from qc_simulator.qc import *
import math
import numpy as np

def build_3qubit_encode_gate():
    """
    Encodes the second and third qubit for 3qubit quantum error correction
    Outputs:
            gate: <Operator>
    """
    I=IdentityGate()
    c_not=CUGate(Not())
    gate=(I % c_not) * (c_not % I)

    return gate


def build_3qubit_ancilla_gate():
    """
    Updates the two ancilla gates for 3qubit quantum error correction
    Outputs:
            <Operator> object
    """
    I=IdentityGate()
    c_not_1i=CUGate(Not(), empty_qw=1)
    c_not_2i=CUGate(Not(), empty_qw=2)
    c_not_3i=CUGate(Not(), empty_qw=3)

    gate=(c_not_2i % I)*(I % c_not_1i % I)*(c_not_3i)*(I % I % c_not_1i)

    return gate

def build_3qubit_correction_gate():
    """
    Corrects a single qubit bit error in 3 qubit quantum error correction
    using information from ancilla qubits.
    Outputs:
            gate: <Operator> object
    """

    rev_c_c_not = build_rev_c_c_not()
    rev_c_c_not_1i = build_rev_c_c_not(empty_qw_target = 1)
    rev_c_c_not_2i = build_rev_c_c_not(empty_qw_target = 2)
    rev_c_not_1i = build_rev_c_not(empty_qw = 1)
    I = IdentityGate()

    gate = rev_c_c_not_2i * (I % rev_c_not_1i % I) * (I % rev_c_c_not_1i)\
    * (I % I % rev_c_not_1i) * (I % I % rev_c_c_not)

    return gate

def build_9qubit_encode_gate():
    """
    Encodes the second to eighth qubit for 9qubit quantum error correction
    Outputs:
            gate: <Operator> object
    """
    c_not_2i=CUGate(Not(), empty_qw=2)
    c_not_5i=CUGate(Not(), empty_qw=5)
    h_gate = Hadamard()
    I=IdentityGate()
    c_not=CUGate(Not())
    c_not_1i=CUGate(Not(), empty_qw=1)

    gate = (c_not_1i % c_not_1i % c_not_1i) * (c_not % I % c_not % I % c_not %I)\
    * (h_gate % I % I % h_gate % I % I % h_gate % I % I) * ( c_not_5i % I % I)\
    * (c_not_2i % I % I % I % I % I)

    return gate


def build_9qubit_ancilla_gate():
    """
    updates the two ancilla gates for 9qubit quantum error correction.
    Outputs:
            gate: <Operator> object
    """

    h_gate =  Hadamard(9)
    N =Not()
    c_not_1i = CUGate(N, empty_qw = 1)
    c_not_2i = CUGate(N, empty_qw = 2)
    c_not_3i = CUGate(N, empty_qw = 3)
    c_not_4i = CUGate(N, empty_qw = 4)
    c_not_5i = CUGate(N, empty_qw = 5)
    c_not_6i = CUGate(N, empty_qw = 6)
    c_not_7i = CUGate(N, empty_qw = 7)
    c_not_8i = CUGate(N, empty_qw = 8)
    I = IdentityGate()

    gate= (h_gate % IdentityGate(2))\
    * (IdentityGate(8) % c_not_1i) * (IdentityGate(7) % c_not_2i)\
    * (IdentityGate(6) % c_not_3i) * (IdentityGate(5) % c_not_4i)\
    * (IdentityGate(4) % c_not_5i) * (IdentityGate(3) % c_not_6i)\
    * (IdentityGate(5) % c_not_3i % I) * (IdentityGate(4) % c_not_4i % I)\
    * (IdentityGate(3) % c_not_5i % I) * (IdentityGate(2) % c_not_6i % I)\
    * (I % c_not_7i % I) * (c_not_8i % I) * (h_gate % IdentityGate(2))

    return gate



def build_9qubit_correction_gate():
    """
    Corrects a single qubit flip error in 9qubit quantum error correction
    using information from ancilla qubits.
    Outputs:
            gate: <Operator> object
    """

    I = IdentityGate()
    z = PhaseShift(np.pi)
    c_c_z_8i = CUGate(z, n_control=2, empty_qw=[8,0])
    c_c_z_4i = CUGate(z, n_control=2, empty_qw=[4,0])
    c_c_z_1i = CUGate(z, n_control=2, empty_qw=[1,0])
    c_z_8i = CUGate(z, n_control=1, empty_qw=8)
    c_z_2i = CUGate(z, n_control=1, empty_qw=2)

    gate = (c_z_8i % I) * (c_c_z_8i) * (IdentityGate(4) % c_c_z_4i)\
    * (IdentityGate(7) % c_z_2i) * (IdentityGate(7) % c_c_z_1i)

    return gate

################################################## Run #######################
if __name__ == '__main__':
#
# I=IdentityGate()
# n=Not()
# reg=(n%I%n%n)* QuantumRegister(4)
# print(reg)
# #z = PhaseShift(np.pi)
# z=Not()
# c_c_z = CUGate(z, n_control=2, empty_qw=[1,0])
# reg=c_c_z*reg
#
#
#
# print(reg)

    # Build 3 Qubit gates
    encode_3_gate=build_3qubit_encode_gate()
    ancilla_3_gate=build_3qubit_ancilla_gate()
    correct_3_gate=build_3qubit_correction_gate()

    # Build 9 Qubit gates
    encode_9_gate=build_9qubit_encode_gate()
    ancilla_9_gate=build_9qubit_ancilla_gate()
    correct_9_gate=build_9qubit_correction_gate()

    # Prepare 1 qubit register
    reg1 = QuantumRegister(1)
    # Prepare 8 qubit encoding register
    reg2 = QuantumRegister(2)
    # Prepare 2 qubit ancilla register
    reg3=QuantumRegister(2)

    print("3 Qubit code and 9 qubit code Quantum Error Correction demonstration")
    print("\n")


    print("3 Qubit code:")
    print("\n")


    print("Single qubit:")
    print(reg1)
    print("\n")
    # Combine register 1 with encoding register
    reg=reg1*reg2

    # Apply encoding gate
    reg= encode_3_gate*reg

    # Print register
    print("Encoded register:")
    print(reg)
    print("\n")

    # Induce a qubit phase flip and print register
    n = Not()
    reg = (n % IdentityGate(1) % IdentityGate(1)) * reg

    print("Qubit phase flip error induced")
    print(reg)
    print("\n")

    # Combine register with ancilla register
    reg=reg*reg3


    # Apply ancilla gate
    reg=ancilla_3_gate*reg

    # Print register
    print("Ancilla gates encoded")
    print(reg)
    print("\n")

    # Apply correction gate
    reg = correct_3_gate * reg

    print("Corrected gate")
    print(reg)
    print("\n")

    print("9 Qubit code:")
    print("\n")
    # Prepare 1 qubit register
    reg1 = QuantumRegister(1)
    # Prepare 8 qubit encoding register
    reg2 = QuantumRegister(8)
    # Prepare 2 qubit ancilla register
    reg3=QuantumRegister(2)


    print("Single qubit:")
    print(reg1)
    print("\n")


    # Combine register 1 with encoding register
    reg=reg1*reg2

    # Apply encoding gate
    reg= encode_9_gate*reg

    # Combine register with ancilla gates
    reg=reg*reg3


    # Print register
    print("Encoded 9 qubit register:")
    print(reg)
    print("\n")

    # Induce a qubit phase flip and print register
    z_gate = PhaseShift(np.pi)
    reg = (IdentityGate(3) % z_gate % IdentityGate(7)) * reg

    print("Qubit phase flip error induced")
    print(reg)
    print("\n")


    # Apply ancilla gate
    reg=ancilla_9_gate*reg

    # Print register
    print("Ancilla gates encoded")
    print(reg)
    print("\n")

    # Apply correction gate
    reg = correct_9_gate * reg

    print("Corrected gate")
    print(reg)
