from qc_simulator.qc import *
from qc_simulator.functions import  *
import numpy as np

class LewisOracle():
    def __init__(self, x):
        self.x = x
    def __getitem__(self, key):
        if key == x:
            return 1
        else:
            return 0

def grovers2():

    # Tag
    k=3

    # Number of qubits
    nqubits=5

    # Initiate the gates
    n_gate_1=Not(1)
    h_gate_n=Hadamard(nqubits)
    h_gate_1=Hadamard(1)
    i_gate_1=Operator(1,base=np.eye(2))

    # Initiate the registers
    reg1=QuantumRegister(nqubits)
    reg2=QuantumRegister(1)
    reg2=n_gate_1*reg2
    reg2=h_gate_1*reg2
    reg1=h_gate_n*reg1
    print(reg1.qubits)

    # Define fk 
    fk=CUGate(not, 
