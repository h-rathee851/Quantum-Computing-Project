import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt

try:
    from src.sparse_matrix import SparseMatrix
except:
    from sparse_matrix import SparseMatrix


class QuantumRegister(SparseMatrix):
    def __init__(self, n_qubits = 1, base_state_ = None):
        self.n_states = 2**n_qubits
        self.n_qubits = n_qubits
        super(QuantumRegister, self).__init__(self.n_states,1)
        if base_state_ is None:
            self.setElement(0, 0, complex(1))
        else:
            if len(base_state_) == self.n_states:
                for i in range(len(base_state_)):
                    if abs(base_state_[i]) != 0:
                        self.setElement(i, 0, complex(base_state_[i]))
                self.normalize()
            else:
                raise ValueError('Length of states is incorrect!')

    def setState(self, state_):
        self.matrix.clear()
        if len(state_) == self.n_states:
            for i in range(len(state_)):
                if abs(state_[i]) != 0:
                    self.setElement(i, 0, complex(state_[i]))
            self.normalize()
        else:
            raise ValueError('Length of base states is incorrect!')

    def measure(self, doPrint=False):
        """
        Collapses the quantum wavefunction in to a possible state.
        :param: (bool) doPrint, True if the measurement of the system is to be
                printed; False otherwise.
        """
        probabilities_ = np.zeros(self.n_states)
        for i in range(self.n_states):
            if (i,0) in self.matrix:
                probabilities_[i] = abs(self.matrix[(i, 0)])**2
            else:
                probabilities_[i] = 0
        state =  int (np.random.choice(self.n_states, p=probabilities_))
        if doPrint:
            print("The measured state is: |"
                + np.binary_repr(state, self.n_qubits) + ">")
        return state

    def __str__(self):
        """
        :return: (str) String representing the quantum register in a terminal
                printable format.
        """
        rep = ''
        for i in range(self.n_states):
            if (i, 0) in self.matrix:
                rep = rep + '({0:+.2f})'.format(self.matrix[(i, 0)]) + "*|"\
                    + np.binary_repr(i, self.n_qubits) + "> "
            else:
                continue
        return rep

    def normalize(self):
        """
        Normalizes the magnitude of the quantum register s.t. the magnitude of
        the register is equal to 1.
        Example:
            [[1],  => [[1/sqrt(2)],
            [1]]      [1/sqrt(2)]]
        """
        norm = 0
        for (i, j) in self.matrix:
            norm += abs(self.matrix[(i, j)])**2
        for (i, j) in self.matrix:
            self.setElement(i, j, (1 / cmath.sqrt(norm)) * self.matrix[i, j])

    def __mul__ (self, other):
        """
        Computes the normalised outer product of the quantum register with
        another matrix or other register.
        :param: (QuantumRegister / SparseMatrix / numpy.array)
        """
        if isinstance(other, QuantumRegister):
            result = QuantumRegister(self.n_qubits + other.n_qubits)
            result.matrix = self.outerProduct(other).matrix
            result.normalize()
            return result
        else:
            raise TypeError('Multiplication not defined between quantum' +
                        ' register and {}.'.format(type(other)))

    def plotRegister(self):
        x_ = []
        y_ = []
        for i in range(self.n_states):
            if (i, 0) in self.matrix:
                x_.append("|" + np.binary_repr(i, self.n_qubits) + ">")
                y_.append(abs(self.matrix[(i, 0)]))
        xpos_ = np.arange(len(x_))
        plt.bar(xpos_, y_)
        plt.xlabel("Qubit states")
        plt.ylabel("Amplitude")
        plt.title("Current quantum register state")
        plt.show()

    def split(self, n_a, n_b):
        if n_a + n_b != self.n_qubits:
            raise ValueError(
                'Number of qubits of subregisters must be '
                'equal to total number of qubits of current register!')
        # Calculate number of base states for each subregister
        n = int(2 ** (n_a))
        k = int(2 ** (n_b))
        a_states_ = []
        b_states_ = []
        for i in range(0, self.n_states, k):
            new_b_states_ = [0] * k
            for j in range(0, k):
                if (i + j, 0) in self.matrix:
                    new_b_states_[j] = self.matrix[(i + j, 0)]
                else:
                    new_b_states_[j] = complex(0)
            normal = norm(np.abs(new_b_states_))
            if normal != 0:
                new_b_states_ = new_b_states_ / normal
            elif normal == 0:
                a_states_.append(0 + 0.j)
                continue
                # print(new_b_states_)
                # print('B states '+ str(b_states_))
            if np.array_equal(b_states_, []):
                b_states_ = new_b_states_
                a_states_.append(normal)
                continue
            elif not np.array_equal(b_states_, new_b_states_):
                # Compare absolute values and then if equal, go through each element trying to
                # find a miltiple. if multiple same, append the multiple to a_states_
                if np.array_equal(np.abs(b_states_), np.abs(new_b_states_)):
                    a_states_holder = new_b_states_[0] / b_states_[0]
                    for i in range(1, len(b_states_)):
                        if a_states_holder == new_b_states_[i] / b_states_[i]:
                            continue
                        elif new_b_states_[i] != 0 and b_states_[i] != 0:
                            raise TypeError('The registers are entangled')
                        else:
                            continue
                    a_states_.append(a_states_holder * normal)
                else:
                    raise TypeError('The registers are entangled')
            else:
                a_states_.append(normal)
                continue
        a = QuantumRegister(n_a,a_states_)
        b = QuantumRegister(n_b,b_states_)
        return(a,b)


def main():
    example = QuantumRegister(n_qubits=4)
    example.setState([0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1])
    example = example
    print(example)
    example.plotRegister()
    print(example.measure())

if __name__ == '__main__':
    main()
