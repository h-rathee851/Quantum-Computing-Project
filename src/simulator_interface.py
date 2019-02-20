import abc

class ISparseMatrix(abc.ABC):
    @abc.abstractmethod
    def setElement(self, i, j, m):
        """
        Sets a cell (i, j) to the value of m within the matrix.
        :param: (int) i, Row number of element to be set.
        :param: (int) j, Column number of element to be set.
        :param: (complex) m, The value of the cell.
        """
        pass

    @abc.abstractmethod
    def getElement(self, i, j):
        """
        Returns the value stored at element (i, j) in the matrix.
        :param: (int) i, Row of desired element.
        :param: (int) j, Column of desired element.
        :return: (complex), The value of element (i, j)
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        """
        :return: (str) String representing the quantum register in a terminal
                printable format.
        """
        pass


@SMatrix.register
class IRegister(SparseMatrix):
    @abc.abstractmethod
    def setState(self, state_):
        """
        ADD FUNCTION DESCRIPTION HERE.
        """
        pass

    @abc.abstractmethod
    def measure(self, doPrint=False):
        """
        Colllapses the quantum superposition in to a possible state.
        :param: (bool) doPrint, True is the measurment of the system is to be
                printed; False otherwise.
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        """
        :return: (str) String representing the quantum register in a terminal
                printable format.
        """
        pass

    @abc.abstractmethod   # THIS MAY NOT BE NEEDED AS IT IS AN INTERNAL FUNCTION.
    def normalize(self):
        """
        Normalizes the magnitude of the quantum register s.t. the magnitude of
        the register is equal to 1.
        Example:
            [[1],  => [[1/sqrt(2)],
            [1]]      [1/sqrt(2)]]
        """
        pass

    @abc.abstractmethod
    def __mul__(self, other):
        """
        Computes the normalised outer product of the quantum register with
        another matrix or other register.
        :param: (QuantumRegister / SparseMatrix / numpy.array)
        """
        pass


@SMatrix.register
class IOperator(SparseMatrix):
    @abc.abstractmethod
    def __mul__(self, rhs):
        """
        :return: (QuantumRegister / Operator) Inner product result. Return type
                depends on the type of the input rhs.
        """
        pass

    @abc.abstractmethod
    def __mod__(self, rhs):
        """
        Calculates the outer product betweeen its self and rhs.
        :param: (Operator) rhs.
        :return: (Operator) Outer product result.
        """
        pass
