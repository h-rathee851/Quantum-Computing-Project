"""
University of Edinburgh, Schoo of Physics and Astronomy
Quantum Computing Project

Shor's algorithm, implemented as a class.
"""


import numpy as np
from qc_simulator.qc import *
from math import gcd

class Shors(Object):
    """
    shors factoring class returns set of factors
    """
    def __init__(self, N, tries=10, accuracy=1):
        """
        N number to be factored
        tries number of tries to find factors
        when creating QRs the calculation is
        bits required to store int((N+1)*accuracy)
        so accuracy can be a float
        """
        self.accuracy = accuracy
        cs = []
        print("shor's algorithum")
        if N%2 == 0:
            cs.extend([2])
            N = int(N/2)
            #print("------------", cs)
        print("input: ",N)
        for i in range(tries):
            c = self.classical(N)
            if c == [0]:
                break
            print(c, cs)
            cs.extend(c)
        cs = np.array(cs)
        cs = cs.flatten()
        cs = np.unique(cs)
        cs = [c for c in cs if c!=1]
        cs = set(cs)
        if cs =={}:
            print("is N a prime")
            self.out = Shors(N,tries,accuracy)
        else:
            self.out = set(cs)

    def classical(self, N):
        """
        clasical component of the algorithum
        """
        if self.check_prime(N):
            print("classical")
            m = np.random.randint(2,N-1)
            #m = 2
            d = gcd(m,N)
            if d!=1:#
                print("Quantum computing required: ")
                return [d]
            else:
                print("Quantum computing required",m," :")
                p = self.find_period(N,m)
                #print("period(p): ",p)
                #print("m**p: ",m**p)
                #print("m**(p/2): ",m**(p/2))
                return self.period_check(N,m,p)
        else:
            print("its a prime")
            return [0]

    def check_prime(self,N):
        """
        check if N is a prime or not
        return true if not prime
        """
        for i in range(2, N):
            if N%i == 0:
                return True
        return False

    def period_check(self, N, m, p):
        """
        checks to see if the period is aceptable
        returns factor if acceptable
        """
        if p%2 != 0:
            print("oops-------1")
            return self.classical(N)
        elif (m**(p/2))%N == 1%N:
            print("oops-------2")
            return self.classical(N)
        elif (m**(p/2))%N == -1%N:
            print("oops------3")
            return self.classical(N)
        else:
            c = [gcd(N, int(m**(p/2)-1)), gcd(N, int(m**(p/2)+1))]
            return c


    def find_period(self, N, m):
        """
        finds period in for N the number to be factored
        and m the base of the powers, f(x) = m**x mod N
        """
        n_qubits = len(format(int((N+1)*self.accuracy),'b'))
        if n_qubits%2!=0:
            n_qubits = n_qubits+1
        print("find period with ", n_qubits, " qubits")
        QR1 = QuantumRegister(n_qubits)
        QR1 = Hadamard(n_qubits)*QR1
        QR2 = QuantumRegister()
        QR = self.fmapping_lazy(QR1, QR2, N, m, n_qubits)
        QFT = self.QFT(n_qubits)
        QR = (QFT%IdentityGate(n_qubits))*QR
        states = QR.base_states
        #print(states)
        print("measureing")
        #c = self.get_p(QR)
        #print(c)
        #c = gcd(QR.measure(), QR.measure())
        #c = gcd(self.mes(QR), self.mes(QR))
        c = self.mes(QR)
        print("measured")
        #print(c)
        return c

    def get_p(self,QR):
        """
        not curently in use
        for taking repeted measurments
        don't this its a good idea
        """
        c = self.mes(QR)
        for i in range(2):
            c = gcd(c, self.mes(QR))
        return c

    def mes(self,QR):
        """
        takes and returns measurment of quantum register QR
        only reurns when the measurment QR is not 1
        """
        c=1
        while c == 1:
            c = QR.measure()
        print(c)
        return c


    def QFT(self, n):
        """
        quantum fourier tansform
        returns quantum fourier tansform matrix for n qubits
        """
        print("QFT")
        H = Hadamard()
        M = H%IdentityGate(n-1)
        for i in range(n-2):
            phi = 2*np.pi/np.power(2,i+2)
            M = (CUGate(PhaseShift(phi),1,i)%IdentityGate(n-2-i))*M
        M = (CUGate(PhaseShift(phi),1,n-2))*M
        for j in range(n-2):
            M = (IdentityGate(j+1)%H%IdentityGate(n-j-2))*M
            for i in range(n-3-j):
                #print(i)
                phi = 2*np.pi/np.power(2,i+2)
                #print(M.matrix.shape)
                M1 = (IdentityGate(j+1)%(CUGate(PhaseShift(phi),1,i))%IdentityGate(n-j-i-3))
                #print(j+1,i,n-j-i-3)
                #print(M1.matrix.shape)
                M = M1*M
            #print("pass")
            phi = 2*np.pi/np.power(2,n-j)
            M1 = (IdentityGate(j+1)%CUGate(PhaseShift(phi),1,n-3-j))
            M = M1*M
        M = (IdentityGate(n-1)%H)*M
        M = SWAPGate(n)*M
        return M


    def fmapping_lazy(self, QR1, QR2, N, m, n_qubits):
        """
        maps|x> and |o> onto |x>|f(x)>
        f(x) = m^x mod N
        returns |x>|f(x)>
        """
        print("lazy mapping")
        #n_qubits = QR1.n_qubits
        n_states = 2**n_qubits
        QR2 = QuantumRegister(n_qubits)
        states = np.zeros(n_states)
        for i in range(n_states):
            x = int(np.mod(m**i, N))
            states[x] = states[x] +1
        QR2.normalise()
        QR2.base_states = states
        QR = QR1*QR2
        return QR
