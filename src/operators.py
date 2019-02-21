#******************************************************************************#
#                                                                              #
# Creates Sparse Matricies which drive from the opperator class.               #
# Contains the quantum gates: I, H, X, Y, Z, R_phi, RX, RY, RZ, CZ, CNOT,      #
# CCNOT, S, T, SWAP, SQUSwap, CSWAP, ISWAP and PSWAP.                          #
#                                                                              #
#******************************************************************************#

import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt
from sparse_matrix import SparseMatrix
from quantum_operator import Operator


class I(Operator):
    """
    Identity operator. Created an identity matrix of size 2*n_qubits and sets
    each element on the major diagonal to 1 while leving all other elements 0.
    """
    def __init__(self, n_qubits: int=1):
        dimension = 2*n_qubits
        base = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    base[i][j] = 1
        super(I, self).__init__(n_qubits, base)


class H(Operator):
    """
    Hadamard gate acts on a single qubit. This gate maps the state,
    |0> to (|0> + |1>) / sqrt(2) and |1> to (|0> - |1>) / sqrt(2), creating a
    super position of the two states. This represesnts a rotation of pi about
    the Z-axis and pi/2 about the Y-axis of a Bloch sphere.
    """
    def __init__(self, n_qubits: int=1):
        base = 1 / np.sqrt(2) * np.array([[1, 1],
                                        [1, -1]])
        super(H, self).__init__(n_qubits, base)


class X(Operator):
    """
    Pauli-X gate acts on a single qubit. It is the quantum equivelent of the
    classical NOT gate. This is equivelent to a rotaation of the Bloch sphere
    through pi radians about the X-axis.
    """
    def __init__(self, n_qubits: int=1):
        base = np.array([[0, 1],
                        [1, 0]])
        super(X, self).__init__(n_qubits, base)


class Y(Operator):
    """
    The Pauli-Y gate acts on a single qubit. It equates to a rotation around the
    Y-axis of the Bloch sphere by pi radians.
    """
    def __init__(self, n_qubits: int=1):
        base = np.array([[0, 1j],
                        [1j, 0]])
        super(Y, self).__init__(n_qubits, base)


class Z(Operator):
    """
    Pauli-Z gate acts on a single qubit. It is equivelent to a rotation about
    the Z-axis of the Bloch sphere by pi radians.
    """
    def __init__(self, n_qubits: int=1):
        base = np.diag([1, -1])
        super(Z, self).__init__(n_qubits, base)


class R_phi(Operator):
    """
    Phase shift gate. This gate acts on a single qubit and rotates though an
    angle phi along the plane of latitude of the Bloch sphere.
    """
    def __init__(self, phi, n_qubits: int=1):
        base = np.diag([1, np.exp(1j * phi)])
        super(R_phi, self).__init__(n_qubits, base)


class RX(Operator):
    def __init__(self, phi, n_qubits: int=1):
        base = np.array([[np.cos(phi / 2.), -1j * np.sin(phi / 2.)],
                        [-1j * np.sin(phi / 2.), np.cos(phi / 2.)]])
        super(RX, self).__init__(n_qubits, base)


class RY(Operator):
    def __init__(self, phi, n_qubits: int=1):
        base = np.array([[np.cos(phi / 2.), -np.sin(phi / 2.)],
                        [np.sin(phi / 2.), np.cos(phi / 2.)]])
        super(RY, self).__init__(n_qubits, base)


class RZ(Operator):
    def __init__(self, phi, n_qubits: int=1):
        base = np.array([[np.cos(phi / 2.) -1j * np.sin(phi / 2.), 0],
                        [0, -1j * np.sin(phi / 2.) + np.cos(phi / 2.)]])
        super(RZ, self).__init__(n_qubits, base)


class CZ(Operator):
    def __init__(self, n_qubits: int=2):
        base = np.diag([1, 1, 1, -1])
        super(CZ, self).__init__(n_qubits, base)


class CNOT(Operator):
    """
    Controlled not gate (or cX gate) which acts on 2 qubits and performs the not
    operation on the second qubit only when the first (control) qubit is |1>.
    """
    def __init__(self, n_qubits: int=2):
        base = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
        super(CNOT, self).__init__(n_qubits, base)


class CCNOT(Operator):
    """
    Toffoli (CCNOT) gate. This is a three quibit gate which performs the NOT
    opperation on the third qubit if states both one and two are in state |1>.
    """
    def __init__(self, n_qubits: int=3):
        base = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  # We should probably come up with a better way of doing this.
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0]])
        super(CCNOT, self).__init__(n_qubits, base)


class S(Operator):
    def __init__(self, n_qubits: int=1):
        base = np.diag([1, 1j])
        super(S, self).__init__(n_qubits, base)


class T(Operator):
    def __init__(self, n_qubits: int=1):
        base = np.diag([1, cmath.exp(1j * np.pi / 4.0)])
        super(T, self).__init__(n_qubits, base)



class SWAP(Operator):
    """
    The Swap gate acts on two qubits and swaps the states of the two input
    qubits.
    Example: |10> -> |01> or |01> -> |10>.
    """
    def __init__(self, n_qubits: int=2):
        base = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
        super(SWAP, self).__init__(n_qubits, base)


class SQUSwap(Operator):
    """
    The SQUSwap gate acts on two qubits and performs a half swap between the
    two qubits.
    qubits.
    """
    def __init__(self, n_qubits: int=2):
        base = np.array([[1, 0, 0, 0],
                        [0, (1/2)*(1+1j), (1/2)*(1-1j), 0],
                        [0, (1/2)*(1-1j), (1/2)*(1+1j), 0],
                        [0, 0, 0, 1]])
        super(SQUSwap, self).__init__(n_qubits, base)



class CSWAP(Operator):
    """
    The Fredikn gate (also CSWAP or cS gate) acts on 3 qubits and performs a
    controlled swap between the second and third qubits if the first qubit is
    is state |1>.
    """
    def __init__(self, n_qubits: int=3):
        base = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  # We should probably come up with a better way of doing this.
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]])
        super(CSWAP, self).__init__(n_qubits, base)


class ISWAP(Operator):
    def __init__(self, n_qubits: int=2):
        base = np.array([1, 0, 0, 0],
                        [0, 0, 1j, 0],
                        [0, 1j, 0, 0],
                        [0, 0, 0, 1])
        super(ISWAP, self).__init__(n_qubits, base)


class PSWAP(Operator):
    def __init__(self, n_qubits: int=2):
        base = np.array([1, 0, 0, 0],
                        [0, 0, np.exp(1j * phi), 0],
                        [0, np.exp(1j * phi), 0, 0],
                        [0, 0, 0, 1])
        super(PSWAP, self).__init__(n_qubits, base)






def test():
    H = Hadamard(1)
    #
    # H_2 = Hadamard(1)
    #
    # print((H_1%H_2))
    # I = Identity()
    # M = H % I
    # print(M)
    r = R_phi(1)
    print(r)

# test()
