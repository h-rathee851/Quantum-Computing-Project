#******************************************************************************#
#                                                                              #
#                         Source code for Grover's                             #
#                            search algorithm.                                 #
#                                                                              #
#******************************************************************************#

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
    """
    All functionality needed to run a simple implementation of Grover's
    algorithm.
    """
    def __init__(self, n_qubits, target_number=None):
        """
        :param: (int) n_qubits: Number of qubits used in the system.
        :param: (int) target_number: A number that represents the type of
                solutions to be found. For example; multiples-of target_number,
                where target_number could be 5 and the Grover search would look
                for multiples of 5.
        """
        self.n_qubits = n_qubits
        self.target_number = target_number
        self.qr = None
        self.D = None
        self.oricle = None
        self.results = [] # List to store collected results.

    def build_quantum_register(self):
        """
        Constructs a quantum register from the n_qubits.
        """
        self.qr = QuantumRegister(self.n_qubits)
        return self.qr

    def print_register(self):
        """
        Prints the current state of the quantum register in a more terminal
        friendly manner.
        """
        print(self.qr)

    def init_register(self):
        """
        ########################################################################
        """
        h = H(self.n_qubits)
        x = X(self.n_qubits)
        h2 = H(self.n_qubits)
        empty_register = QuantumRegister(self.n_qubits)
        self.qr = h * self.qr
        return self.qr

    def init_reflection_matrix(self):
        """
        Constructs a reflection matrix which can rotate the desired state
        through pi radians within a Bloch sphere.
        """
        h = H(self.n_qubits)
        r = G_Phase(self.n_qubits)
        self.D = h * r * h
        return self.D

    def gen_oracle(self, selection=1):
        """
        Sets up the oracle needed to solve for particular states. There are
        three implementations here which one is implemented depends on the
        input 'selection'.
        :param: (int)/(str) selection: choosed the type of search problem to be
                implamented.
        If selection is '5' then a basic example oracle is implemented that
        finds all states which are multiples of 5.
        If selection is '1' or 'multiples' or 'multiples-of' then an oracle
        which solves for multiples of a given target_number.
        If selection is '2' or 'powers' or 'powers-of' then an oracle is
        generated which solves for numbers which are target_number**n where n is
        an integer.
        """
        if selection == 5:
            self.oracle = Oracle(self.n_qubits)
        elif (selection == '1' or selection == 'multiples'
            or selection == 'multiples-of'):
            self.oracle = GeneralOracle(self.target_number, 1, self.n_qubits)
        elif (selection == '2' or selection == 'powers'
            or selection == 'powers-of'):
            self.oracle = GeneralOracle(self.target_number, 2, self.n_qubits)
        else:
            print("Problem selection not valid. Try either: multiples-of." +
            "Where selection is '1' or 'multiples' or 'multiples-of'. Or;" +
            "powers-of. Where selection is '2' or 'powers' or 'powers-of'.")
            sys.exit()
        return self.oracle

    def run(self, k=None):
        """
        Runs Grover's algorithm for a number of runs calculated from the number
        of target states. In the case where there is an unknow number of target
        states the system is run for the worst case scenareo number of runs.
        :param: (int) k: Number of target stares.
        """
        if k == 1:
            runs = round(np.sin(np.pi/8) * np.sqrt(self.n_qubits))
        elif k > 1:
            runs = round(np.sqrt(self.n_qubits))
        else:
            runs = round(np.sqrt(self.n_qubits))

        for i in range(int(runs)):
            self.qr = self.oracle * self.qr
            self.qr = self.D * self.qr
            self.qr.normalize()
        result = self.qr.measure()
        self.results.append(result)
        return result

    def plot_results(self):
        """
        Plots a histogram of frequency of measurement to a given state against
        possible states. The regions with greater amplitude are likley to be the
        target state(s).
        """
        bins = np.arange(2**self.n_qubits) - 0.5
        x = np.array(self.results)
        plt.hist(self.results, bins=bins, rwidth=0.95)
        plt.ylabel("State measurement frequency")
        plt.xlabel("States")
        plt.title("Measurement frequency plot")
        plt.show()

    def print_results(self, itterations):
        """
        Prints the results of each measurement to the terminal along with the
        number of times that that state has been measured.
        :param: (int) itterations: Total number of measurements taken.
        """
        r = np.sort(self.results)
        unique, counts = np.unique(r, return_counts=True)
        d = dict(zip(unique, counts))
        print("\n\nOf %d measurements:\n" % itterations)
        for key, value in sorted(d.items()):
            if value == 1:
                print("%4d, measured %4d time. Selected %3.1f%% of the time"
                 % (key, value, (value/itterations)*100))
            else:
                print("%4d, measured %4d times. Selected %3.1f%% of the time."
                 % (key, value, (value/itterations)*100))

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
