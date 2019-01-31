# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:29:09 2018

@author: Lewis
"""
import unittest
import numpy as np
import math
from qc_simulator.qc_testing import *
from qc_simulator.functions import *

class QCTesting(unittest.TestCase):
    def test_mul_Hadamard(self):
        reg1 = QuantumRegister(1)
        H = Hadamard()
        applied = np.asmatrix((H*reg1).base_states)
        H_test = 1/math.sqrt(2)*np.matrix([[1,1],[1,-1]])
        expected = np.dot(H_test, reg1.base_states)
        np.testing.assert_almost_equal(expected.tolist(),applied.tolist())


    def test_mul_CHadamard(self):
        reg1 = QuantumRegister(2)
        CH = CUGate(Hadamard(),1)
        result = (CH*reg1).base_states
        expected= np.array((1,0,0,0))
        self.assertEqual(result.tolist(),expected.tolist())
        
        
    def test_oracle(self):
        O = Oracle(10,x=5)
        reg = QuantumRegister(10)
        result = O*reg
        self.assertEqual(result.base_states[5], -1 * reg.base_states[5])
        for i in range(reg.base_states.size):
            if i != 5:
                self.assertEqual(result.base_states[i], reg.base_states[i])
                
                
    def test_phase_shift(self):
        P = PhaseShift(math.pi)
        H = Hadamard(1)
        reg = H*QuantumRegister(1)
        regbase_states = reg.base_states
        result = P*reg
        
        gate = np.array([[1,0],[0,-1]])
        expected = np.dot(gate,regbase_states)  
        np.testing.assert_almost_equal(expected.tolist(),result.base_states.tolist())
        
    def test_deutsch(self):
        o1 = Operator(1, np.array([[1,0],[0,1]]))
        self.assertEqual(deutsch(o1),0)
        o2 = Operator(1,np.array([[1,0],[0,-1]]))
        self.assertEqual(deutsch(o2),1)
       
    def test_grover(self):
        o = Oracle(10,x=5)
        result = grover_search(o)
        self.assertEqual(5,result)
     
    def test_adder(self):
        a = QuantumRegister(1)
        b = QuantumRegister(1)
        result = quantumAdder(a,b)
        print(result.base_states)
        self.assertEqual(np.array([1,0]),result.base_states)
        
    def testControlV(self):
        reg = QuantumRegister(1)
        reg.base_states = np.array([1, 1])
        result = ControlV(1)*reg
#        expected = reg
        expected.base_states[-1] *= 1j
        np.testing.assert_almost_equal(expected.base_states.tolist(),result.base_states.tolist())
        
    def test_operator_tensor(self):
        H = Hadamard()
        result = H%H
        expected = 0.5*np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])
        np.testing.assert_almost_equal(result.matrix.toarray().tolist(),expected.tolist())
    
    def test_settingbase_states(self):
        a = GetQuBitofValue(np.pi/4,np.pi/4)
        expected = (1/np.sqrt(2))*np.array([1,1])
        result = np.absolute(a.base_states)
        np.testing.assert_almost_equal(expected,result)
        
    def test_plot_register(self):
        reg = QuantumRegister(4)
        print(QuantumRegister.__dict__)
        H = Hadamard(4)
        reg = H*reg
        
        reg.plot_register(False)
        
        
        
unittest.main()