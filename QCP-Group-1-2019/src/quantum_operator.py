# Defining the structure and operations which Operators can be built from.

import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt
try:
    from src.sparse_matrix import SparseMatrix
    from src.quantum_register import QuantumRegister
except:
    from sparse_matrix import SparseMatrix
    from quantum_register import QuantumRegister

class Operator(SparseMatrix):

    """
    Operator class inherits from SparceMatrix
    """
    def __init__(self, n_qubits : int=1, base = np.zeros((2,2))):

        """
        Class constructor

        Inputs:
               n_qubits <int>: Number of qubits the oparator acts on
               base <np.array>: Matrix representation for the operators
        """

        if n_qubits <= 0 :
            raise ValueError('Operator must operate on at least 1 qubit!')
        self.n_qubits = n_qubits
        self.size = 2 ** n_qubits
        if self.size < len(base):
            raise ValueError("Operator cannot act on the specified number" +
                            "of qubits.")
        act_qubits = int(np.log2(len(base)))
        base_matrix = SparseMatrix(*[len(base)]*2)
        for i in range(0, len(base)):
            for j in range(0, len(base)):
                if base[i][j] != 0:
                    base_matrix.setElement(i, j, complex(base[i][j]))
                else:
                    continue
        for i in range(0, n_qubits, act_qubits):
            if i == 0:
                result = base_matrix
                continue
            result = result.outerProduct(base_matrix)
        super(Operator, self).__init__(self.size,self.size)
        self.matrix = result.matrix

    def __mul__(self, rhs):
        """
        :return: (QuantumRegister / Operator) Inner product result. Return type
                depends on the type of the input rhs.
        """
        if isinstance(rhs, QuantumRegister):
            result = QuantumRegister(n_qubits = self.n_qubits)
        elif isinstance(rhs, Operator):
            result = Operator(n_qubits = self.n_qubits)
        else :
            " Raise type error if the right type isn't provided"
            raise TypeError('Multiplication not defined for Operator' +
                            ' and {}.'.format(type(rhs)))
        if rhs.n_qubits != self.n_qubits:
                raise ValueError(
                    'Number of states do not correspnd: rhs.n_qubits = {},' +
                    ' lhs.n_qubits = {}'.format(rhs.n_qubits, self.n_qubits))
        result.matrix = self.innerProduct(rhs).matrix
        return result

    def __mod__(self, rhs):
        """
        Calculates the outer product betweeen its self and rhs.
        :param: (Operator) rhs.
        :return: (Operator) Outer product result.
        """
        if isinstance(rhs, Operator):
            result = Operator(self.n_qubits + rhs.n_qubits)
            result.matrix = self.outerProduct(rhs).matrix
            return result
        else:
            raise TypeError('Operation not defined between operator and ' +
                            '{}.'.format(type(rhs)))
