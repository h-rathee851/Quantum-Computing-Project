# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:12:25 2018

@author: John
"""
import numpy as np

def print_ket(QR):
    """
    prints ket notation in binary form
    """
    N = len(QR)
    n = np.log2(N)
    for i in range(N):
        x = np.binary_repr(i)
        print(str(QR[i])+'|',x,'>')


def print_ket_int(QR):
    """
    prints ket notation in int form
    """
    N = len(QR)
    n = np.log2(N)
    for i in range(N):
        print(str(QR[i])+'|',i,'>')

class QFT():
    def __init__(self, n):
        if n!=0:
            self.n = n
            self.ipower = 2*np.pi/n
            self.w = np.exp((2*np.pi*1j)/n, dtype=np.complex_)
            #print(self.w)
        else:
            print("dimension input 'n' can not be zero")


    def CPhasegate(self, phi, control, QR):
        """
        control phase, applies exp(phi*1j) to all elements in QR
        with 1 in control the q_bit place
        example control = 0, 3 bits, aplies to |001>, |011>, |101>, |111>
        """
        N = len(QR)
        x = np.power(2, control-1)
        for i in range(N):
            if i%2 == 0:
                for j in range(x):
                    QR[i*j] = QR[i*j]*np.exp(phi*1j, dtype=np.complex_)
        return QR

    def R_phi(self, phi, target, QR):
        """
        control phase, applying exp(phi*1j) to target
        """
        QR[target] = QR[target]*np.exp(phi*1j, dtype=np.complex_)
        return QR

    def R_phase(self, phase, target, QR):
        """
        control phase, applying phase to target
        """
        QR[target] = QR[target]*phase
        return QR
    
    def apply_U(self, QR, U, m, n=-1):#untested
        """
        applys 2by2 matrix onto specifiec places in a QR
        does not construct matrix, to be used only for large QR
        would require diffrent matrix multiplication that holds the order of
        application then applies in that order when used
        """
        if n == -1:
            n = m + 1
        QR[m] = QR[m]*U[0,0] + QR[n]*U[0,1]
        QR[m+1] = QR[m]*U[1,0] + QR[n]*U[1,1]
        return QR


    def matrix(self, n):
        """
        matrix for QFT
        """
        m = np.ones((n,n), dtype=np.complex_)
        for i in range(n):
            for j in range(n):
                print((i*j)%n)
                m[i,j] = np.power(self.w, (i*j)%n,dtype=np.complex_)
        print(m)
        return m

    #req funtioning hadamard and controled phase gate
    def Circuit(self, n, QR): # working on this
        """
        applies QFT using base gates
        """
        for i in range(n):
            hadamard()
            for j in range(n-i-1):
                w = np.power(self.w,j,dtype=np.complex_)
                QR = self.R_phase(w, j, QR)
        #need gate to swap order of qbits, i.e. 100 <-> 001 switch
        #may drop due to needing swap
        return QR #may need renormalised
    
class shors():
    """
    lots of diffrent variations to implement, at the moment doing a simple one
    will do a more complex later possibly the 2n+3 q-bits circute
    """
#    def __init__(self,QR):
#        n = len(QR)
#        self.shor(n, QR)

    def error(x):
        """
        for when U is given no function
        """
        print("no function given to U")
        return x

    def func(self, x, a, N):
        """
        function that will be used
        """
        return (np.power(x,a)%N)

    def U(self, *x, f=err):
        """
        aplies funtion given onto x
        """
        return f(*x)
    
    def Shor_Simple(self, n, QR):
        """
        using matrix, not base gates
        """
        print("simple")

    def Shor_Circuit(self, n, QR):
        """
        using base gates
        """
        QR = self.Circuit(n, QR)

#below for testing

#qr=np.array([15,81,6,51,6,5], dtype=np.complex_)

#print_ket(np.array([1,2,3,1]))
#print_ket_int(np.array([1,2,3,1]))
#class shor():
#    print(1)
#
a = QFT(4)
b = shors_simple()
c = b.U(2)

print(c)
#print(a.apply_U(qr, np.array([[2,2],[2,2]], dtype=np.complex_),0))
#print(a.R_target(-np.pi, 0, qr))

#for i in range(5):
 #   QFT(i)
print("done")