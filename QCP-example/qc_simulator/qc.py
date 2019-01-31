"""
University of Edinbugh, School of Physics and Astronomy
Quantum Computing Project

Quantum Computer Simulator: Implementations of quantum register and quantum
logic gate.

Authors: Andreas Malekos, Gabriel Hoogervorst, Lewis Lappin, John Harbottle,
Wesley Shao Zhonghua, Huw Haigh
"""

import numpy as np
from numpy.linalg import norm
from scipy.sparse import identity as sparse_identity
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix, kron
from math import pi
import matplotlib.pyplot as plt
from copy import deepcopy

#import abstract classes
from qc_simulator.qc_interface import *

class QuantumRegister(IQuantumRegister):
    """
    Quantum register class. The quantum register is saved as a complex
    numpy array. Each element of the array is the amplitude of the
    corresponding state, eg. the first element is the first state, the second
    element is the second state.
    """


    def __init__(self, n_qubits: int = 1, base_states = None):
        """
        Class constructor
        Inputs:
                n_qubits: <int> number of qubits in quantum register
                isempty: <bool> parameter that tells whether the quantum register should be empty.
                Set to False by default.
        """
        # Check that qubits is bigger than 0
        if n_qubits <= 0 or (not isinstance(n_qubits, int)):
            raise ValueError(
            'Quantum register must have at least 1 qubit of integer type!')

        self.n_states = int(2 ** n_qubits)
        self.n_qubits = n_qubits
        # If isempty = False, initialise in ground state
        if base_states == None:
            self.base_states = np.zeros(self.n_states, dtype=complex)
            self.base_states[0] = 1.0
        else:
            # Check if length is correct
            if len(base_states) != self.n_states:
                raise ValueError(
                'Length of base states is incorrect!'
                )

            self.base_states = np.array(base_states, dtype=complex)
            #Normalise
            self.normalise()

    def measure(self):
        """
        Make a measurement. Square all the amplitudes and choose a random state.
        Outputs an integer representing the state measured (decimal system).
        Outputs:
                state: <int> integer that corresponds to the number of the state measured, in decimal format.
        """
        # Calculate probabilities
        probabilities = np.zeros(self.n_states)
        for i in range(self.n_states):
            probabilities[i] = norm(self.base_states[i]) ** 2

        # Choose a random state
        n = int(self.n_states)
        state =  int (np.random.choice(n, p=probabilities) )

        return state

    def __mul__(self, other):
        """
        Overrides multiplication operator to define tensor product between two quantum registers.
        Inputs:
                other: <QuantumRegister> right hand side quantum register
        Outputs:
                qmr_result: <QuantumRegister> resulting quantum register.

        Raises error if 'other' is not of the right type.
        """

        # Check if other is of the right tyoe
        if isinstance(other, QuantumRegister):
            # Result is tensor product of the qubits in each state
            temp_result = np.kron(self.base_states, other.base_states)

            # Result has to be normalized
            result_normalized = temp_result / norm(temp_result)

            # Create quantum register object for result
            qmr_result = QuantumRegister(self.n_qubits + other.n_qubits)
            qmr_result.base_states = result_normalized

            return qmr_result
        else:
            raise TypeError('Multiplication not defined between quantum register and {}.'.format(type(other)))

    def __str__(self):
        """
        Overrides str method to print out quantum register in braket notation
        Outpiuts:
                rep : <str> reply
        """
        base_states = self.base_states
        l = len(base_states)
        n_qubits = self.n_qubits
        if base_states[0] != 0:
            rep = '({0:+.2f})'.format(base_states[0]) + "*|" + np.binary_repr(0, n_qubits) + "> "
        else:
            rep = ''

        for i in range(1, l):
            if base_states[i] == 0:
                continue
            rep = rep + '({0:+.2f})'.format(base_states[i]) + "*|" + np.binary_repr(i, n_qubits) + "> "

        return rep



    def split(self, n_a, n_b):
        """
        Assuming the quantum register is not an entagled state, splits into two
        subregisters given the length of each.
        Inputs:
                n: <int> Number of qubits of first subregister
                k: <int> Number of qubits of second subregister
        Outputs:
                a, b: (<QuantumRegister>, <QuantumRegister>) Tuple containing the two sub_registers

        Raises error if n_a and n_b are not equal to the n_qubits.
        User must make sure that the quantum register is not entangled. Nonsensical
        results will occur otherwise. 
        """
        # Check if n_a + n_b = total number of qubits in register
        if n_a + n_b != self.n_qubits:
            raise ValueError(
            'Number of qubits of subregisters must be '
            'equal to total number of qubits of current register!')

        # Extract base states
        base_states = self.base_states

        # Calculate number of base states for each subregister
        n = int(2 ** (n_a))
        k = int(2 ** (n_b))

        # Create array containing multiples of subregister b
        multiples_of_b = np.zeros((n,k), dtype=complex)
        multiples_of_b = np.reshape(base_states, (n,k))

        # Calculate the square root of the sum of the squares
        sum_of_bs = np.sum( abs(multiples_of_b), 1)

        # Calculate the square root of the sum squared
        c = np.sqrt(np.sum( np.square( norm(sum_of_bs) ) ) )

        # Divide norm_of_bs by c
        a_states = sum_of_bs/c

        # If the square root of the sum of the norms squared isn't 1, then
        # the quantum register is entangled, raise error and stop operation
        test_norm = np.sqrt( np.sum( np.square(norm(a_states) ) ) )

        #  Extract sub register b
        b_states = multiples_of_b[0,:]/a_states[0]

        # Create two new quantum registers and put them in a tuple
        a = QuantumRegister(n_a)
        a.base_states = a_states

        b = QuantumRegister(n_b)
        b.base_states = b_states

        return (a, b)


    def normalise(self):
        """
        Normalise coefficients of qubits array.
        """
        # Add tolerance to remove extremely small floating point calculation errors
        tol = 10 ** (-8)
        filter = abs(self.base_states) >= tol
        self.base_states = self.base_states * filter

        base_states_normalised = self.base_states / norm(self.base_states)

        self.base_states = base_states_normalised

    def plot_register(self, show=True):
        """
        Produce bar graph of quantum register.
        Inputs:
                show: <bool> Flag that if set to true, shows the bar graph.
        Outputs:
                ax: axis handle object.
        """
        fig, ax = plt.subplots()
        rects1 = ax.bar(np.arange(2**self.n_qubits),np.absolute(self.base_states))
        if show:
            plt.show()
        return ax

    def plot_bloch(self):
        """
        Creates a bloch sphere of the quantum register.
        """
        
        
        b = Bloch()
        objs = []
        for i in range(self.n_qubits):
            obj = Qobj(self.base_states[2*i:2*i+2])
            b.add_states(obj)
            objs.append(obj)
        #b.add_states(objs)
        b.show()

class Operator(IOperator):
    """
    Class that defines a quantum mechanical operator. Implments abstract class
    OperatorAbstract. The operator is stored as a square sparse matrix.
    """

    def __init__(self, n_qubits: int = 1, base=np.zeros((2, 2))):
        """
         Class constructor
         Inputs:
                n_qubits: <int> Number of qubits operator operates on
                base: <np.array> Base matrix

        Raises error if n_qubits is less than or equal to 0.
        User must make sure that n_qubits is <int>.
        User must make sure that the size of base is consistant, i.e. a 2 by 2
        numpy array.
        """
        # Check if number of qubits is correct
        if n_qubits <= 0 :
            raise ValueError('Operator must operate on at least 1 qubit!')

        self.n_qubits = n_qubits
        self.size = 2 ** n_qubits
        self.matrix = self.__create_sparse_matrix(self.n_qubits, base)

    def __create_sparse_matrix(self, n_qubits, base):
        """
        Create matrix by taking successive tensor producs between for the total
        number of qubits.
        Inputs:
                n_qubits: <int> Number of qubits operator operates on
                base: <np.array> Base matrix
        Outputs:
                <csc_matrix> Sparse matrix (csc format)
        """
        base_complex = np.array(base, dtype=complex)
        result = lil_matrix(base_complex)

        if n_qubits == 1:
            result = csc_matrix(result)

            return result
        else:
            for i in range(n_qubits - 1):
                result = kron(result, base)

            result = csc_matrix(result)
            return result

    def __mul__(self, rhs):
        """
        Overrides multiplication operator and defined the multiplication between
        two operators and an operator and a quantum register.
        Inputs:
                rhs: <QuantumRegister> / <Operator> Right hand side, can be either
                operator or quantum register
        Outputs:
                result:<QuantumRegister> / <Operator> if rhs is of type Operator then
                return Operator. If it's of type QuantumRegister, then return a quantum
                register object.

        In case where rhs is <QuantumRegister> raises error if its number of qubits
        do not correspond to the number of qubits of the Operator.

        In case where rhis is <Operator> raises error if it is not of the same size.

        User needs to make sure that rhs is of the right type.
        """
        if isinstance(rhs, QuantumRegister):
            # Apply operator to quantum register
            # check if number of states is the same
            if rhs.n_qubits != self.n_qubits:
                raise ValueError(
                    'Number of states do not correspnd: rhs.n_qubits = {}, lhs.n_qubits = {}'.format(rhs.n_qubits,
                                                                                                     self.n_qubits))

            # Otherwise return a new quantum register
            result = QuantumRegister(rhs.n_qubits)

            # Calculate result. Check if matrix is sparse or not first. If sparse
            # use special sparse dot product csc_matrix.dot
            result.base_states = self.matrix.dot(rhs.base_states.transpose())

            # Normalise result
            result.normalise()
            return result

        if isinstance(rhs, Operator):
            """
            Matrix multiplication between the two operators
            """
            if rhs.size != self.size:
                raise ValueError(
                    'Operators must of of the same size: rhs.size = {} lhs.size = {} '.format(rhs.size, self.size))

            # Otherwise take dot product of
            result = Operator(self.n_qubits)
            result.matrix = self.matrix.dot(rhs.matrix)
            return result

        else :
            " Raise type error if the right type isn't provided"
            raise TypeError(
                'Multiplication not defined for Operator and {}.'.format(type(rhs))
            )



    def __mod__(self, other):
        """
        Overrides "%" operator to define tensor product between two operators.
        Inputs:
                other: <Operator> Right hand side
        Outputs:
                result: <Operator> Result of tensor product
        """

        # Tensor product between the two operators
        if isinstance(other, Operator):
            result = Operator(self.n_qubits + other.n_qubits)
            result.matrix = csc_matrix(kron(self.matrix, other.matrix))
            return result
        else:
            raise TypeError(
                'Operation not defined between operator and {}.'.format(type(other))
            )

    def __str__(self):
        """
        Provides method to pring out operator.
        Outputs:
                rep: <String> String that corresponds the __str__() method of the
                numpy array.
        """
        return self.matrix.toarray().__str__()

    def dag(self):
        """
        Computes hermitian transpose of operator.
        Outputs:
                herm_transpse: <Operator> Hermitian transpose of operator
        """

        herm_transpose = Operator(self.n_qubits)
        herm_transpose.matrix = self.matrix.getH()

        return herm_transpose

class Hadamard(Operator):
    """
    Class that defines hadamard gate. This class extends the Operator class.
    """

    def __init__(self, n_qubits: int =1):
        # Define "base" hadamard matrix for one qubit and correponding sparse matrix
        base = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        super(Hadamard, self).__init__(n_qubits, base)

class PhaseShift(Operator):
    """
    Class that implements phase shift gate
    """

    def __init__(self, phi, n_qubits: int =1):
        """
        Class constructor.
        Inputs:
                phi: <float> angle in radians
                n_qubits: <int> number of qubits
        """
        base = np.array([[1, 0], [0, np.exp(1j * phi)]])
        super(PhaseShift, self).__init__(n_qubits, base)

class Not(Operator):
    """
    Class that implements not gate.
    """

    def __init__(self, n_qubits=1):
        base = np.array([[0, 1], [1, 0]])
        super(Not, self).__init__(n_qubits, base)

class CUGate(Operator):
    """
    Class that implements a controlled U gate.
    """

    def __init__(self, base, n_control=1, empty_qw=0):
        """
        Class constructor.
        Inputs:
                base: <Operator> base Operator U
                n_control: <int> number of control qubits
                empty_qw: <int>\ <list>: Number of empty lines between control
                qubits and target qubits.
        If there are no empty lines, leave equal to 0. If there are, empty_qw
        must be a list with n_control-1 elements, each indicating the number of
        empty lines between control-control and finally control-target eg [1,0,1]
        """
        if not isinstance(empty_qw, int):
            if len(empty_qw) != n_control:
                raise ValueError('Number of empty lines must correctly specified!')
        elif     n_control !=1 and empty_qw!=0:
                raise ValueError('Number of empty lines must be correctly specified!')

        self.n_control = n_control
        self.n_qubits = 1 + self.n_control + np.sum(empty_qw)
        self.size = 2 ** (self.n_qubits)
        self.empty_qw = empty_qw
        self.matrix = self.__create_sparse_matrix(base)

    def __create_sparse_matrix(self, base):
        """
        Creates sparse matrix according to how many target qubits we have.
        Matrix is constructed using the 'lil' format, which is better for
        incremental construction of sparse matrices and is then converted
        to 'csc' format, which is better for operations between matrices
        Inputs:
                base: <Operator> base operator U
        Outputs:
                <csc_matrix> Sparse matrix representing operator
        """

        # Create sparse hadamard matrix
        base_matrix = lil_matrix(base.matrix)

        # Create full sparse identity matrix
        sparse_matrix = sparse_identity(self.size, dtype=complex, format='lil')

        if np.sum(self.empty_qw) == 0:
            # "Put" dense hadamard matrix in sparse matrix
            target_states = 2
            sub_matrix_index = self.size - target_states
            sparse_matrix[sub_matrix_index:, sub_matrix_index:] = base_matrix

            # Convert to csc format and return
            return csc_matrix(sparse_matrix)
        else:
            # Find indices of contro qubits
            control_qubit_indices = self.__find_control_qubits()

            # Loop over the columns and check to see if the corresponding states
            # have all the control states set to 1
            for i in range(int(self.size/2), self.size,2):
                # Extract binary version of number
                bin_i_str = np.binary_repr(i, self.n_qubits)

                # Convert to numpy array
                bin_i = np.array([int(x) for x in bin_i_str])

                # Return indexes of elements equal to 1
                indices_of_ones = np.flatnonzero(bin_i)

                # Check if control qubits are set to 1
                control_qubit_check = np.isin(control_qubit_indices, indices_of_ones)
                if np.all(control_qubit_check):
                    # If true then then put base matrix onto diagonal
                    sparse_matrix[i:i+2, i:i+2] = base_matrix

            return csc_matrix(sparse_matrix)

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
            control_qubit_indices[1:] = control_qubit_indices[1:] + self.empty_qw[:-1]

            return control_qubit_indices

class IdentityGate(Operator):
    """
    Class that implements identity operator.
    """
    def __init__(self, n_qubits = 1):
        super(IdentityGate, self).__init__(n_qubits, base=np.eye(2,2))

class fGate(Operator):
    """
    Class that implements an f-Gate, Uf. The action of Uf is defined as follows:
    Uf*|x>*|y> = |x>*|(y+f(x))%2>
    """
    def __init__(self, f, n):
        """
        Class constructor:
        Inputs:
                f: <type 'function'> callable function f defined from {0,1}^n -> {0,1}
                n: <int> number of states n acts on
        """
        self.f = f
        self.n_qubits = n + 1
        self.size = 2**(self.n_qubits)
        self.matrix = self.__f_matrix()

    def __f_matrix(self):
        """
        Constructs a numpy matrix that corresponds to the function
        evaluation. The matrix is then converted to a sparse array.
        Outputs:
                <csc_matrix> Sparse matrix containing the matrix represnetation
        of the operator.
        """
        matrix_full = np.eye(self.size, self.size)
        n = int(self.size/2)
        f = self.f

        for i in range(n):
            # Loop over the rows 2 at a time and exchange only if f(x)
            # returns 1.
            if f(i) == 1:
                temp = deepcopy(matrix_full[2*i,:])
                temp2 = deepcopy(matrix_full[2*i + 1,:])
                matrix_full[2*i,:] = temp2
                matrix_full[2*i + 1, : ] = temp

        return csc_matrix(matrix_full)

class SWAPGate(Operator):
    """
    Class that implements a SWAP gate acting on an n qubit register.
    """
    def __init__(self, n_qubits):
        # Check that correct numbr of qubits has been entered
        if n_qubits <= 1:
            raise ValueError('SWAP Gate must operate on at least 2 qubits!')

        self.n_qubits = n_qubits
        self.size = int(2**(n_qubits))
        self.matrix = self.__create_sparse_matrix()

    def __create_sparse_matrix(self):
        """
        Creates sparse matrix for SWAP Gate
        Outputs:
                matrix: <csc_matri> Matrix representing SWAP gate
        """
        n_qubits = self.n_qubits
        size = self.size

        # Create empty numpy array
        dense_matrix = np.zeros((size,size))

        # Loop for every row
        for i in range(size):
            state_binary = np.binary_repr(i, n_qubits)

            # Flip string and convert to integer
            state_flipped = state_binary[::-1]
            k = int(state_flipped, 2)

            #Assign relevant matrix element to 1
            dense_matrix[i,k] = 1

        # Convert dense matrix to csc_matrix
        return csc_matrix(dense_matrix)
