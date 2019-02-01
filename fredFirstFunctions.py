"""
Generate the basic Quantum Gates
"""

"""

a,b,c,d are the real & imaginary coordinates of the 0 & 1 basis the qubit
ie qubit 1 = x = a + ib
qubuit 2 = y = c + id

"""

import numpy as np
import math
import cmath

def identity(a,b,c,d):
    a = a
    b = b
    c = c
    d = d
    return a,b,c,d


def Hadamard(a,b,c,d):
    a = (1/math.sqrt(2))*(a+c)
    b = (1/math.sqrt(2))*(b+d)
    c = (1/math.sqrt(2))*(a+c)
    d = (1/math.sqrt(2))*(b+d)
    return a,b,c,d

def phase(a,b,c,d,angle):
    a = a
    b = b
    c = c*np.cos(angle) - d*np.sin(angle)
    d = d*np.cos(angle) + c*np.sin(angle)
    return a,b
"""
cnot is a two-qubit gate, with each represented by real and imaginary parts
"""
def cnot(a,b,c,d,e,f,g,h):
    a = a
    b = b
    c = c
    d = d
    e = g
    f = h
    g = e
    h = f
    return a,b,c,d,e,f,g,h

def cphase(a,b,c,d,e,f,g,h,angle):
    a,b,c,d = identity(a,b,c,d)
    e,f,g,h = phase(e,f,g,h,angle)
    return a,b,c,d,e,f,g,h

"""
Now have a universal set of gates above
"""

def V(a,b,c,d):
    a = a
    b = b
    c = -d
    d = c
    return a,b,c,d

def Vdagger(a,b,c,d):
    for i in range(3):
        a,b,c,d = V(a,b,c,d)
        
    return a,b,c,d

def cV(a,b,c,d,e,f,g,h):
    a,b,c,d = identity(a,b,c,d)
    e,f,g,h = V(e,f,g,h)
    return a,b,c,d,e,f,g,h

def cVdagger(a,b,c,d,e,f,g,h):
    a,b,c,d = identity(a,b,c,d)
    e,f,g,h = Vdagger(e,f,g,h)
    return a,b,c,d,e,f,g,h
    

"""
Now attempt to construct the Toffoli Gate
"""

def toffoli(a,b,c,d,e,f,g,h,i,j,k,l):
    i,j,k,l = Hadamard(i,j,k,l)
    e,f,g,h,i,j,k,l = cV(e,f,g,h,i,j,k,l)
    a,b,c,d,e,f,g,h = cnot(a,b,c,d,e,f,g,h)
    e,f,g,h,i,j,k,l = cVdagger(e,f,g,h,i,j,k,l)
    a,b,c,d,e,f,g,h = cnot(a,b,c,d,e,f,g,h)
    a,b,c,d,i,j,k,l = cV(a,b,c,d,i,j,k,l)
    i,j,k,l = Hadamard(i,j,k,l)
    return a,b,c,d,e,f,g,h,i,j,k,l

"""
And the Quantum Adder
"""

def qAdder(a,b,c,d,e,f,g,h,i,j,k,l):
    """
    e,f,g,h gives the sum
    i,j,k,l gives the carry
    """
     a,b,c,d,e,f,g,h,i,j,k,l = toffoli(a,b,c,d,e,f,g,h,i,j,k,l)
     a,b,c,d,e,f,g,h = cnot(a,b,c,d,e,f,g,h)
     return a,b,c,d,e,f,g,h,i,j,k,l
    
  



