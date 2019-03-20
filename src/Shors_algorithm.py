#******************************************************************************#
#                                                                              #
#                         Source code for Shor's                               #
#                          factoring algorithm.                                #
#                                                                              #
#******************************************************************************#

import random
import math
import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt
try:
    from src.sparse_matrix import SparseMatrix
    from src.quantum_register import QuantumRegister
    from src.quantum_operator import Operator
    from src.operators import *
    from src.QFT import *
    from src.quantum_shor import *
except:
    from sparse_matrix import SparseMatrix
    from quantum_register import QuantumRegister
    from quantum_operator import Operator
    from operators import *
    from QFT import *
    from quantum_shor import *

def continued_fraction(y, Q):
    """
    Calculates continued fraction expansion for a rational fraction. It uses the Euclidean algorithm.
    The output is a classical notation for continued fraction [a0, a1, a2, a3, ..., an].
    """
    a = []
    integer, remainder1 = divmod(y, Q)
    a.append(integer)
    integer, remainder2 = divmod(Q, remainder1)
    a.append(integer)
    
    if remainder2 != 0:
        while True:
            integer, remainder3 = divmod(remainder1, remainder2)
            if remainder3 == 0:
                a.append(integer)
                break
            a.append(integer)
            remainder1 = remainder2
            remainder2 = remainder3

    return a

def all_Shor(N, t_qubits):
    """
    Main function to carry the full simulation of Shor's algorithm.
        :param: (int) N: Integer to be factored.
        :param: (int) t_qubits: number of qubits used in each register -
                the greater number of qubits, the higher accuracy of results.

    """
    l = 0 #when l == 1, then the while loop is stopped - factors are found;
    repeat = 0 #for checking if the quantum subroutine was repeated
    
    while l == 0: #when no non-trivial factor of N is found
        #classical preprocessing
        m = random.randint(2, N-1) #pick a random integer m, such that  1 < m < N
        Q = 2**t_qubits
        b = math.gcd(m, N) #find the greatest common divisor of m and N

        if b != 1:
            #b is a nontrivial factor of N
            l = 2
            f = N/b
            print("No need to proceed to phase estimation.")
            print(str(b) + " and " + str(int(f)) + " are non-trivial factors of " + str(N) + ".")

        else: #if b == 1
            #Quantum part
            #do period-finding using phase estimation
            if repeat == 0:
                #do it only once at the first time quantum measurement is performed
                inv_qft = invQFT(t_qubits) #initialising inverse QFT

            QR3, second_reg_vals_ = UaGate(N, m, t_qubits)
            QR = measure_second_reg(N, m, t_qubits, second_reg_vals_)
            ft = inv_qft * QR #apply inverse quantum Fourier transform
            ft.plotRegister() #show a plot of amplitudes for each qubit state 

            y = ft.measure()
            print("Measurement on the first register.")
            print(y)
            #classical postprocessing
            #Estimate y/Q in lower terms using continued fraction expansion. This will yield d/r estimation.

            if y != 0:
                #continued fraction expansion
                a = continued_fraction(y, Q)

                #find all candidates for integer d
                d = []
                d.append(a[0]) #d0 = a0
                d.append(1 + a[0]*a[1]) #d1 = 1 + a0*a1
                for i in range(2, len(a)):
                    d.append(a[i]*d[i-1]+d[i-2])

                #find all possible candidates for period r
                period = []
                period.append(1)
                period.append(a[1])
                for i in range(2, len(a)):
                    period.append(a[i]*period[i-1]+period[i-2])

                #finding more suitable r candidates
                canditate_no = []
                for i in range(1, len(period)):
                    #r and d have to be co-prime
                    div = period[i]/d[i]
                    if math.gcd(period[i], d[i]) == 1 and div != 0 and div != 1:
                        canditate_no.append(period[i])

                if canditate_no: #if the list of r candidates is not empty
                    #testing r candidates
                    for k in range(len(canditate_no)):
                        r = canditate_no[k]
                        if r % 2 == 0 and (m**(r/2)) % N != -1: #good candidate
                            l = 1 #stop the big while loop; factors can be found now
                            break

                else: #if no r candidates were found
                    l = 0
                    repeat = 1 #quantum measurement might be repeated
                    print("No suitable r candidates were found. The algorithm is rerun.")


            else: #the measurement y is 0
                l = 0
                repeat = 1 #quantum measurement might be repeated
                print("The measurement y is 0. The algorithm is rerun.")

        if l == 1: #final stage: finding non-trivial factors

            mod = (m**(r/2) % N) + 1
            gcd = math.gcd(int(mod), N)

            if gcd == 1 or gcd == N:
                mod = (m**(r/2) % N) - 1
                gcd = math.gcd(int(mod), N)

            if gcd == 1 or gcd == N:
                l = 0
                print("Trivial factors are found. The algorithm is rerun.")
            else:
                f = int(N/gcd)
                print(str(gcd) + " and " + str(f) + " are non-trivial factors of " + str(N) + ".")


"""
testing components of Shor's algorithm:
"""
#test continued fraction expansion
def test_cont_frac():
    y = 427 #measurement from the first register
    Q = 512 #number of superposition states
    a = continued_fraction(y, Q)
    print(a)

#test all Shor's algorithm
if __name__ == '__main__':
    N = 15 #integer to be factored
    t_qubits = 9 #number of qubits used in each register in the quantum part
    all_Shor(N, t_qubits)
