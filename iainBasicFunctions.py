"""
Basic functions for quantum computing.
Aiming to cover:
  Hadamard,
  Phase shift,
  C2-Not,

"""
import numpy as np

def matrix_magnatude(matrix):
    return np.linalg.norm(matrix)

def hadamard_gate(qubit):
    norm_fact = 1 / np.sqrt(2)
    matrix = np.array([[1, 1],
                      [1, -1]])
    had = norm_fact * matrix
    return np.dot(had, qubit)

def c_not_gate(control_qubit, target_qubit):
    matrix = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
    input = np.concatenate((control_qubit, target_qubit), axis=None)
    out = np.dot(matrix, input)
    print(out)
    return out

def X_gate(qubit):
    """
    Bit flip (Not gate)
    """
    matrix = np.array([[0, 1],
                       [1, 0]])
    return np.dot(matrix, qubit)

def phase_flip(qubit):
    """
    Z-Gate / Phase-flip
    """
    matrix = np.array([[1, 0],
                       [0,-1]])
    return np.dot(matrix, qubit)

def T_gate(qubit, theta):
    matrix = np.array([[1, 0],
                        [0, np.exp(1j * theta)]])
    return np.dot(matrix, qubit)

def c_V_gate(control_qubit, target_qubit):
    matrix = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1j]])
    input = np.concatenate((control_qubit, target_qubit), axis=None)
    out = np.dot(matrix, input)
    print(out)
    return out  # As with c-not not sure this is correct

def c_2_not(control_1, control_2, target):
    pass

def hadamard_test():
    zero = np.array([[1.],
                     [0.]])
    one = np.array([[0.],
                    [1.]])
    had = hadamard_gate(one)
    mag = matrix_magnatude(had)
    print("One on Hadamard:")
    print(had)
    print("Magnitude of matrix: %lf" % mag)

def c_not_test():
    zero = np.array([[1.],
                     [0.]])
    one =  np.array([[0.],
                     [1.]])
    c_not_gate(zero, zero)  # Not sure this is right


# hadamard_test()
c_not_test()


############################################################################

def outer_product_matrix(u, v):
    vt = np.transpose(v)
    return np.dot(u, vt)

def inner_product(u, v):
    return np.dot(u, v)
