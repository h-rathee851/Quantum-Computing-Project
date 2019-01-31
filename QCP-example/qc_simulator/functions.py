# -*- coding: utf-8 -*-
"""
University of Edinburgh, School of Physics and Astronomy
Quantum Comuting Project

Helper functions containing allowing for the construction of various types
of quantum circuits.
"""
#!/usr/bin/env python3

from qc_simulator.qc import *
import numpy as np
import math



def oracle_single_tag(n, tag):
    """
    Function that builds a quantum circuit representing an oragle that tags
    a single state.
    Inputs:
            n: <int> number of qubits the Operator operates on
            tag: <int> tagged state
    Outputs:
            oracle_gate: <Operator> oracle gate
    """

    n_qubits=n+1

    # Convert tag to binary string
    bina_str=np.binary_repr(tag, n)
    # Convert binary string to list of integers
    binalist=[int(s) for s in bina_str]
    # Reverse list as operators are applied in opposite order
    binalist=binalist[::-1]

    # Initiate base gates
    not_gate = Not()
    I = Operator(base=np.eye(2,2))

    # Initiate the prep gate with if statement
    '''
    if binalist[0]==1:
        prep_gate=I
    elif binalist[0]==0:
        prep_gate=not_gate
    '''
    prep_gate=I
    # For loop to create whole gate
    for i in binalist:
        if i==1:
            prep_gate= I % prep_gate
        if i==0:
            prep_gate= not_gate % prep_gate

    cn_not = CUGate(not_gate, n_qubits-1)

    oracle_gate=prep_gate*cn_not*prep_gate
    return oracle_gate


    # Define middle column of gates
    gates[num_of_gates-1] = IdentityGate(n_qubits-3) % c_c_z

    # Fill out the rest of the array
    gates[num_of_gates: ] = np.flip(gates[:num_of_gates-1], axis=0)

    # Complete gate is the multiplication of everything inside the array
    cnz_gate = np.prod(gates)

    return cnz_gate


def build_c_c_not(empty_qw_control=0, empty_qw_target=0):
    """
    Builds a toffoli gate, given the number of I operators between the second control and the target qubit from the
    first control. By default these distances are set to 0 and 0 respectively.
    Inputs:
            empty_qw_control: <int> Number of empty quantum wires between first
            and second control qubits
            empty_qw_target: <int>  Number of empty quantum wires between
            second control and target qubits
    Outputs:
            toffoli: <Operator> toffoli gate (Operator Object)
    """

    # Initialise basis gates
    h_gate = Hadamard()
    I = IdentityGate()
    I_target = IdentityGate(empty_qw_target + 1)
    I_control = IdentityGate(empty_qw_control + 1)
    I_total = IdentityGate(empty_qw_target + empty_qw_control + 2)

    v_gate = PhaseShift(np.pi / 2)
    c_v_short = CUGate(v_gate, empty_qw=empty_qw_target)
    c_v_long = CUGate(v_gate, empty_qw=empty_qw_target+empty_qw_control + 1)

    c_not = CUGate(Not(), empty_qw=empty_qw_control)
    v3 = v_gate * v_gate * v_gate
    c_v3 = CUGate(v3, empty_qw=empty_qw_target)

    # Build circuit
    toffoli = (I_total % h_gate) * c_v_long * (c_not % I_target) * (I_control % c_v3) \
       * (c_not % I_target) * (I_control % c_v_short) * (I_total % h_gate)

    return toffoli


def build_rev_c_c_not(empty_qw_control=0, empty_qw_target=0):
    """
    Builds a reverse toffoli gate, given the number of I operators between the second control and the target qubit from the
    first control. By default these distances are set to 0 and 0 respectively.
    Inputs:
            empty_qw_control: <int> Number of empty quantum wires between first
            and second control qubits
            empty_qw_target: <int>  Number of empty quantum wires between
            second control and target qubits
    Outputs:
            toffoli: <Operator> Reversed toffoli gate (Operator Object)
    """

    # Initialise basis gates
    h_gate = Hadamard()
    I = IdentityGate()
    I_target = IdentityGate(empty_qw_target + 1)
    I_control = IdentityGate(empty_qw_control + 1)
    I_total = IdentityGate(empty_qw_target + empty_qw_control + 2)

    v_gate = PhaseShift(np.pi / 2)
    c_v_short = CUGate(v_gate, empty_qw=empty_qw_target)
    c_v_long = CUGate(v_gate, empty_qw=empty_qw_target+empty_qw_control + 1)

    c_not = CUGate(Not(), empty_qw=empty_qw_control)
    v3 = v_gate * v_gate * v_gate
    c_v3 = CUGate(v3, empty_qw=empty_qw_target)

    # Build circuit

    if empty_qw_control == 0:
        gate = (h_gate % I_total) * c_v_long  * (I_target % h_gate % h_gate)\
        * (I_target % c_not) * (I_target % h_gate % h_gate) * (c_v3 % I_control)\
        * (I_target % h_gate %  h_gate) * (I_target % c_not)\
        * (I_target % h_gate %  h_gate) * (c_v_short % I_control) * (h_gate % I_total)
    else:

        gate = (h_gate % I_total) * c_v_long  * (I_target % h_gate % IdentityGate(empty_qw_control) % h_gate)\
        * (I_target % c_not) * (I_target % h_gate % IdentityGate(empty_qw_control) % h_gate) * (c_v3 % I_control)\
        * (I_target % h_gate % IdentityGate(empty_qw_control) % h_gate) * (I_target % c_not)\
        * (I_target % h_gate % IdentityGate(empty_qw_control) % h_gate) * (c_v_short % I_control) * (h_gate % I_total)

    return gate



def build_rev_c_not(empty_qw=0):
    '''
    Builds a reverse c not gate
    Inputs:
            empty_qw: <int> Number of qubits between the control and target
    Outputs:
            gate: <Operator> Inverted not gate
    '''


    h_gate = Hadamard()
    c_not = CUGate(Not(), empty_qw=empty_qw)

    if empty_qw == 0:
        gate = (h_gate % h_gate) * c_not * (h_gate % h_gate)

    else:
        I = IdentityGate(n_qubits=empty_qw)
        gate = (h_gate % I % h_gate) * c_not * (h_gate % I % h_gate)

    return gate
