#******************************************************************************#
#                                                                              #
# Creates Sparse Matricies which derive from the opperator class.               #
# Contains the quantum gates: I, H, X, Y, Z, R_phi, RX, RY, RZ, CZ, CNOT,      #
# CCNOT, S, T, SWAP, SQUSwap, CSWAP, ISWAP and PSWAP.                          #
#                                                                              #
#******************************************************************************#

import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt
try:
    from src.sparse_matrix import SparseMatrix
    from src.quantum_operator import Operator
    from src.quantum_register import QuantumRegister
except:
    from sparse_matrix import SparseMatrix
    from quantum_operator import Operator
    from quantum_register import QuantumRegister


class I(Operator):
    """
    Identity operator. Created an identity matrix of size 2*n_qubits and sets
    each element on the major diagonal to 1 while leaving all other elements 0.
    """
    def __init__(self, n_qubits: int=1):
        dimension = 2**n_qubits
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
    super position of the two states. This represents a rotation of pi about
    the Z-axis and pi/2 about the Y-axis of a Bloch sphere.

    If more than one qubits is selected in the argument,
    then a matrix is generated to apply a Hadamard gate individually to all of the qubits
    """
    def __init__(self, n_qubits: int=1):
        base = 1 / np.sqrt(2) * np.array([[1, 1],
                                        [1, -1]])
        super(H, self).__init__(n_qubits, base)


class X(Operator):
    """
    Pauli-X gate acts on a single qubit. It is the quantum equivalent of the
    classical NOT gate. This is equivalent to a rotation of the Bloch sphere
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
    Pauli-Z gate acts on a single qubit. It is equivalent to a rotation about
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
    """
    Rotation about X gate. This acts on a single qubit and rotates
    through an angle phi perpendicular to the X axis of the Bloch sphere.
    """
    def __init__(self, phi, n_qubits: int=1):
        base = np.array([[np.cos(phi / 2.), -1j * np.sin(phi / 2.)],
                        [-1j * np.sin(phi / 2.), np.cos(phi / 2.)]])
        super(RX, self).__init__(n_qubits, base)


class RY(Operator):
    """
    Rotation about Y gate. This acts on a single qubit and rotates
    through an angle phi perpendicular to the Y axis of the Bloch sphere.
    """
    def __init__(self, phi, n_qubits: int=1):
        base = np.array([[np.cos(phi / 2.), -np.sin(phi / 2.)],
                        [np.sin(phi / 2.), np.cos(phi / 2.)]])
        super(RY, self).__init__(n_qubits, base)


class RZ(Operator):
    """
    Rotation about Z gate. This acts on a single qubit and rotates
    through an angle phi perpendicular to the Z axis of the Bloch sphere.
    """
    def __init__(self, phi, n_qubits: int=1):
        base = np.array([[np.cos(phi / 2.) -1j * np.sin(phi / 2.), 0],
                        [0, -1j * np.sin(phi / 2.) + np.cos(phi / 2.)]])
        super(RZ, self).__init__(n_qubits, base)


class CZ(Operator):
    """
    The Controlled-Z gate, which acts on 2 qubits,
    performing the Z transformation on the second qubit
    only when the first is in state |1>.
    """
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
    Toffoli (CCNOT) gate. This is a three qubit gate which performs the NOT
    operation on the third qubit if states both one and two are in state |1>.
    """
    def __init__(self, n_qubits: int=3):
        base = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0]])
        super(CCNOT, self).__init__(n_qubits, base)


class S(Operator):
    """
    The Phase (pi/2) rotation gate. Rotates a single qubit by pi/2 radians
    perpendicular to the real axis
    """
    def __init__(self, n_qubits: int=1):
        base = np.diag([1, 1j])
        super(S, self).__init__(n_qubits, base)


class T(Operator):
    """
    The Phase (pi/4) rotation gate. Rotates a single qubit by pi/4 radians
    perpendicular to the real axis
    """
    def __init__(self, n_qubits: int=1):
        base = np.diag([1, cmath.exp(1j * np.pi / 4.0)])
        super(T, self).__init__(n_qubits, base)


'''
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
'''

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
        base = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]])
        super(CSWAP, self).__init__(n_qubits, base)


class ISWAP(Operator):
    """
    ISWAP gate performs the conventional SWAP
    and performs a pi/2 rotation in the imaginary plane
    """
    def __init__(self, n_qubits: int=2):
        base = np.array([1, 0, 0, 0],
                        [0, 0, 1j, 0],
                        [0, 1j, 0, 0],
                        [0, 0, 0, 1])
        super(ISWAP, self).__init__(n_qubits, base)


class PSWAP(Operator):
    """
    PSWAP gate performs the conventional SWAP
    then rotates by phi in the imaginary plane
    """
    def __init__(self, phi, n_qubits: int=2):
        base = np.array([1, 0, 0, 0],
                        [0, 0, np.exp(1j * phi), 0],
                        [0, np.exp(1j * phi), 0, 0],
                        [0, 0, 0, 1])
        super(PSWAP, self).__init__(n_qubits, base)


class CUGate(Operator):

    def __init__(self,base,n_control = 1, empty_qw = 0,reverse = False):

        if not isinstance(empty_qw, int):
            if len(empty_qw) != n_control:
                raise ValueError('Number of empty lines must correctly specified!')
        elif     n_control !=1 and empty_qw!=0:
                raise ValueError('Number of empty lines must be correctly specified!')

        self.n_control = n_control
        self.n_qubits = base.n_qubits + self.n_control + np.sum(empty_qw)
        self.size = 2 ** (self.n_qubits)
        empty_qw = np.cumsum(empty_qw)
        self.empty_qw = empty_qw
        super(CUGate, self).__init__(self.n_qubits)
        self.matrix = self.__create_sparse_matrix(base)
        if reverse:
            self.matrix = self.__reverse()


    def __create_sparse_matrix(self, base):
        result = SparseMatrix(self.size, self.size)
        for i in range(self.size):
            result.setElement(i, i, complex(1))
        if np.sum(self.empty_qw) == 0:
            # "Put" dense hadamard matrix in sparse matrix
            sub_matrix_index = self.size - base.size
            for i in range(base.size):
                for j in range(base.size):
                    result.setElement(sub_matrix_index + i, sub_matrix_index
                                    + j, complex(0))
                    if (i, j) in base.matrix:
                        result.setElement(sub_matrix_index + i, sub_matrix_index
                                        + j, base.matrix[(i, j)])
            return result.matrix
        else:
            # Find indices of contro qubits
            control_qubit_indices = self.__find_control_qubits()
            # Loop over the columns and check to see if the corresponding states
            # have all the control states set to 1
            for m in range(int(self.size / 2), self.size, base.size):
                # Extract binary version of number
                bin_i_str = np.binary_repr(m, self.n_qubits)
                # Convert to numpy array
                bin_i = np.array([int(x) for x in bin_i_str])
                # Return indexes of elements equal to 1
                indices_of_ones = np.flatnonzero(bin_i)
                # Check if control qubits are set to 1
                control_qubit_check = np.isin(control_qubit_indices,
                                                indices_of_ones)
                if np.all(control_qubit_check):
                    # If true then then put base matrix onto diagonal
                    for i in range(base.size):
                        for j in range(base.size):
                            result.setElement(m + i, m + j, complex(0))
                            if (i, j) in base.matrix:
                                result.setElement(m + i, m + j,
                                                base.matrix[(i, j)])
            return result.matrix

    def __find_control_qubits(self):
        """
        Returns an array containing the index of the control qubits
        Outputs:
                control_qubit_indices: <np.array> Inidces of control qubits
        """
        if self.n_control == 1:
            return np.array([0])
        else:
            control_qubit_indices = np.arange(self.n_control)
            # Exclude the item of self.empty_qw because it's the number of empty
            # lines between the last control and first target qubit
            control_qubit_indices[1:] = (control_qubit_indices[1:]
                                        + self.empty_qw[:-1])
            return control_qubit_indices

    def __reverse(self):
        sw = SWAP(self.n_qubits)
        result = sw * self * sw
        return result.matrix


class SWAP(Operator):
    """
    The Swap gate acts on two qubits and swaps the states of the two input
    qubits.
    Example: |10> -> |01> or |01> -> |10>.
    """
    def __init__(self, n_qubits: int=2 , empty_qw=None):
        # Check that correct numbr of qubits has been entered
        if n_qubits <= 1:
            raise ValueError('SWAP Gate must operate on at least 2 qubits!')
        self.n_qubits = n_qubits
        self.size = int(2**(n_qubits))
        if empty_qw is not None and len(empty_qw) > n_qubits - 1:
            raise ValueError('Number of space between swap qubits must \
                            correctly specified!')
        if empty_qw is None:
            empty_qw = [0] * (n_qubits - 1)
        self.empty_qw = empty_qw
        base = self.__create_matrix()
        super(SWAP, self).__init__(self.n_qubits,base)

    def __create_matrix(self):
        """
        Creates sparse matrix for SWAP Gate
        Outputs:
                matrix: <csc_matri> Matrix representing SWAP gate
        """
        n_qubits = self.n_qubits
        size = self.size
        # Create empty numpy array
        dense_matrix = np.zeros((size, size))
        swap_indices = np.arange(n_qubits - np.sum(self.empty_qw))
        self.empty_qw = np.cumsum(self.empty_qw)
        swap_indices[1:] = swap_indices[1:] + self.empty_qw
        # Loop for every row
        for i in range(size):
            state_binary = np.binary_repr(i, n_qubits)
            bits_to_flip = []
            bits_not_flipped = []
            for m in range(self.n_qubits):
                if m in swap_indices:
                    bits_to_flip.append(state_binary[m])
                else:
                    bits_not_flipped.append(state_binary[m])
            # Flip string and convert to integer
            bits_flipped = bits_to_flip[::-1]
            i_1 = iter(bits_flipped)
            i_2 = iter(bits_not_flipped)
            state_flipped = ''
            for m in range(self.n_qubits):
                if m in swap_indices:
                    state_flipped = state_flipped + next(i_1)
                else:
                    state_flipped = state_flipped + next(i_2)
            k = int(state_flipped, 2)
            #Assign relevant matrix element to 1
            dense_matrix[i, k] = 1
        # Convert dense matrix to csc_matrix
        return dense_matrix


def test():
    h = H(2)
    qr = QuantumRegister(2)
    # #
    # # H_2 = Hadamard(1)
    # #
    # # print((H_1%H_2))
    # i = *I()
    M = h * qr
    print(M)

    # r = R_phi(1)
    # print(r)

if __name__ == '__main__':
    test()
