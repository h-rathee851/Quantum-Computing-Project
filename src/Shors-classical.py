#Shor's algorithm
from random import *
import math

def quantum_Shor(m, N): #includes quantum fourier transform

    """
    Pseudocode for the quantum part:
    Have two registers.
    L is the number of qubits required for number N.
    Have Q (number of superposition states), such that Q = 2^L and N^2 =< Q <2N^2
    (the smallest possible power of 2)
    1. Set all qubits from the first register to zero.
    2. Set all qubits from the second register to 1.
    2. Apply Hadamard gate to all the qubits from the first register.
    3. Construct f(x) = m^(x), mod N, as a quantum function.
       (r is a period to be found -> f(x+r) = f(x)). It uses a second register.

    4. Apply the quantum Fourier transform on the first register.
    5. Apply the measurement on the first register. The measured value is y.
    """
    return y, Q

def continued_fraction(y, Q): #for a rational fraction
    a = []
    integer, remainder1 = divmod(y, Q)
    a.append(integer)
    integer, remainder2 = divmod(Q, remainder1)
    a.append(integer)

    while True:
        integer, remainder3 = divmod(remainder1, remainder2)
        if remainder3 == 0:
            break
        a.append(integer)
        remainder1 = remainder2
        remainder2 = remainder3

    return a

def post_classical(y, Q, m, N): #classical post-processing (after the quantum part)

    #continued fraction expansion
    a = continued_fraction(y, Q)

    #find all candidates for integer d
    d = []
    print(a)
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
                l = 1 #stop the big while loop
                break

    #should I include testing the multiples of r candidates or is it too much?
    #when should I pick a random number m again?
    return r

def all_Shor(N):
    l = 0 #when l == 1, then the while loop is stopped - factors are found;
    #when l == 2, quantum period-finding is not required
    while l == 0: #when no non-trivial factor of N is found

        m = randint(1, N-1) #pick a random integer m, such that  1 < m < N
        b = math.gcd(m, N) #find the greatest common divisor of m and N

        if b != 1:
            #b is a nontrivial factor of N
            l = 2
            print(str(b) + " is a non-trivial factor of " + str(N) + ".")

        else: #if x == 1
            #do quantum period-finding
            #measured obtained after applying QFT
            y, Q = quantum_Shor(m, N)
            #Estimate y/Q in lower terms using continued fraction expansion. This will yield d/r estimation.
            #this method is from https://qudev.phys.ethz.ch/content/QSIT15/Shors%20Algorithm.pdf
            #algorithm for finding continued fraction expansion comes from Nielsen textbook

            r = post_classical(y, Q, m, N)


    if l == 1:
        mod1 = (m**(r/2) + 1) % N
        mod2 = (m**(r/2) - 1) % N
        gcd1 = math.gcd(int(mod1), N)
        gcd2 = math.gcd(int(mod2), N)
        print(str(gcd1) + " and " + str(gcd2) + " are non-trivial factors of " + str(N) + ".")


"""
testing components of Shor's algorithm:
"""
#test continued fraction expansion
def test_cont_frac():
    y = 427 #measurement from the first register
    Q = 512 #number of superposition states
    a = continued_fraction(y, Q)
    print(a)

def test_post_processing():
    y = 427 #measurement from the first register
    Q = 512 #number of superposition states
    m = 11 #from modular function f(x) = m**x
    N = 21 #integer to be factored
    r = post_classical(y, Q, m, N)
    print(r)

if __name__ == '__main__':
    test_post_processing()
