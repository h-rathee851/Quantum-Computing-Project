import numpy as np
from numpy.linalg import norm
import cmath
import math
import matplotlib.pyplot as plt
import os

try:
    from src.sparse_matrix import SparseMatrix
    from src.quantum_register import QuantumRegister
    from src.quantum_operator import Operator
    from src.operators import *
    from src.oracle import *
    from src.grover_phase import *
except:
    from sparse_matrix import SparseMatrix
    from quantum_register import QuantumRegister
    from quantum_operator import Operator
    from operators import *
    from oracle import *
    from grover_phase import *


class Grover:
    def __init__(self, n_qubits, target_number=None):
        self.n_qubits = n_qubits
        self.target_number = target_number
        self.qr = None
        self.D = None
        self.oricle = None
        self.results = []

    def build_quantum_register(self):
        self.qr = QuantumRegister(self.n_qubits)
        return self.qr

    def print_register(self):
        print(self.qr)

    def init_register(self):
        h = H(self.n_qubits)
        x = X(self.n_qubits)
        h2 = H(self.n_qubits)
        empty_register = QuantumRegister(self.n_qubits)
        self.qr = h * self.qr
        return self.qr

    def init_reflection_matrix(self):
        h = H(self.n_qubits)
        c = G_Phase(self.n_qubits)
        self.D = h * c * h
        return self.D

    def gen_oracle(self, selection=1):
        if selection == 5:
            self.oracle = Oracle(self.n_qubits)
        elif (selection == '1' or selection == 'multiples'
            or selection == 'multiples-of'):
            self.oracle = GeneralOracle(self.target_number, 1, self.n_qubits)
        elif (selection == '2' or selection == 'powers'
            or selection == 'powers-of'):
            self.oracle = GeneralOracle(self.target_number, 2, self.n_qubits)

        return self.oracle

    def run(self, k):
        # k is the number of tagged states
        # runs = round(math.sqrt(self.n_qubits/k))
        # runs = round( ((math.pi / 4) * math.sqrt(k)) * 2**(self.n_qubits / 2))  # / or *
        # runs = round((math.pi / 4) * math.sqrt(2**self.n_qubits))
        runs = 10* round(math.sqrt(2**self.n_qubits))
        # print(runs)
        # runs = 100
        for i in range(runs):
            self.qr = self.oracle * self.qr
            self.qr = self.D * self.qr
            self.qr.normalize()
        result = self.qr.measure()
        self.results.append(result)
        return result

    def plot_results(self):
        bins = np.arange(2**self.n_qubits) - 0.5
        x = np.array(self.results)
        plt.hist(self.results, bins=bins, rwidth=0.95)
        plt.ylabel("State measurement frequency")
        plt.xlabel("States")
        plt.title("Measurement frequency plot")
        plt.show()


################################################################################
# Class test functions.
################################################################################

def runG():
    g = Grover(3)
    g.build_quantum_register()
    g.init_register()
    g.init_reflection_matrix()
    g.gen_oracle()
    return g.run(3)

def plot():
    x = []
    for i in range(0,1000):
        x.append(runG())
    bins = np.arange(0, max(x) + 1.5) - 0.5
    x = np.array(x)
    plt.hist(x)
    plt.show()

if __name__ == '__main__':
    # test = QuantumRegister(3,[1,1,1,1,1,1,1,1])
    # o = Oracle(3)
    # print(o*test)
    plot()
