"""
University of Edinbugh, School of Physics and Astronomy
Quantum Computing Project

Quantum Computer Simulator: Definition of interfaces for quantum register
and quantum logic gate 

Authors: Andreas Malekos, Gabriel Hoogervorst, Lewis Lappin, John Harbottle,
Wesley Shao Zhonghua, Huw Haigh
"""

from abc import ABC, abstractclassmethod

class IQuantumRegister(ABC):


    @abstractclassmethod
    def measure(self):
        """
        Measure that defined measurement of quantum register
        """
        pass

    @abstractclassmethod
    def __mul__(self, other: 'IQuantumRegister')-> 'IQuantumRegister' :
        """
        Define multiplication between quantum registers
        """
        pass

    @abstractclassmethod
    def __str__(self):
        """
        Define print method for the quantum register
        """

class IOperator(ABC):

    @abstractclassmethod
    def __mul__(self, rhs):
        """
        Define multiplication between operators, and optionally, between operators and
        quantum registers.
        """
        pass

    @abstractclassmethod
    def __mod__(self, other: 'IOperator') -> 'IOperator' :
        """
        Define tensor product between operators
        """
        pass

    @abstractclassmethod
    def __str__(self):
        """
        Define print method for operator objectself.
        """
        pass
