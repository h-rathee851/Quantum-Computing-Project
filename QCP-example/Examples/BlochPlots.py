# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 19:12:52 2018

@author: Lewis
"""
import math
from qc_simulator.qc import *

H = Hadamard()
reg = QuantumRegister()
N = Not()
P = PhaseShift(math.pi/2)


reg.plot_bloch()
regH = H*reg
regP = P*regH
regH.plot_bloch()
regP.plot_bloch()
print("not")
(N*reg).plot_bloch()