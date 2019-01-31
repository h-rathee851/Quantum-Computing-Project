"""
File containing failed ideas.
"""

def build_nc_not(n):
    """
    Builds n controleld not gate
    :param n: number of control qubits
    :return: cn_gate, controlled not gate
    """

    # Iniate the three base gates
    not_gate = Not()
    c_c_not=build_c_c_not()
    c_not=CUGate(not_gate)

    # Initialise total number of quibts
    n_qubits = n+1


    # Two cases, n is even and n odd
    # Num will be the number of gates necessary
    # If odd
    if (n_qubits%2)==1:
        num_of_gates = n_qubits-2
    if (n_qubits%2)==0:
        num_of_gates = n_qubits-1

    # Initiate the gates list
    gates = np.empty(2 * num_of_gates - 1, dtype=Operator)

    # Define first column of gates
    gates[0] = c_c_not % IdentityGate(n_qubits-3)

    # Construct n_not gate in for Loop
    for i in range(1, num_of_gates-1):

        # Check if we are on an even or odd step
        if num_of_gates%2 == 0:
            num_of_i_above = 2 * (i-1)
            num_of_i_below = n_qubits - num_of_i_above - 3

            I_above = IdentityGate(num_of_i_above)
            I_below = IdentityGate(num_of_i_below)

            gates[i] = I_above % c_c_not % I_below
        else:
            num_of_i_above = 2 * i
            num_of_i_below = n_qubits - num_of_i_above - 1

            I_above = IdentityGate(num_of_i_above)
            I_below = IdentityGate(num_of_i_below)

            gates[i] = I_above % not_gate % I_below




    # Define middle column of gates
    gates[num_of_gates-1] = IdentityGate(n_qubits-3) % c_c_not

    # Fill out the rest of the array
    gates[num_of_gates: ] = np.flip(gates[:num_of_gates-1], axis=0)

    # Complete gate is the multiplication of everything inside the array
    cn_gate = np.prod(gates)

    return cn_gate

def build_nc_z(n):
    """
    Builds an n controlled not gate
    Total number of qubits is therefore n+1
    """
    # Iniate the three base gates
    z_gate = PhaseShift(np.pi)
    c_c_z = CUGate(z_gate, 2)
    not_gate = Not()
    c_c_not=build_c_c_not()
    c_not=CUGate(not_gate)
    #I = Operator(base=np.eye(2,2))

    # Initialise total number of quibts
    n_qubits = n+1


    # Two cases, n is even and n odd
    # Num will be the number of gates necessary
    # If odd
    if (n_qubits%2)==1:
        num_of_gates = n_qubits-2
    if (n_qubits%2)==0:
        num_of_gates = n_qubits-1

    # Initiate the gates list
    gates = np.empty(2 * num_of_gates - 1, dtype=Operator)

    # Define first column of gates
    gates[0] = c_c_not % IdentityGate(n_qubits-3)

    # Construct n_not gate in for Loop
    for i in range(1, num_of_gates-1):

        # Check if we are on an even or odd step
        if num_of_gates%2 == 0:
            num_of_i_above = 2 * (i-1)
            num_of_i_below = n_qubits - num_of_i_above - 3

            I_above = IdentityGate(num_of_i_above)
            I_below = IdentityGate(num_of_i_below)

            gates[i] = I_above % c_c_not % I_below
        else:
            num_of_i_above = 2 * i
            num_of_i_below = n_qubits - num_of_i_above - 1

            I_above = IdentityGate(num_of_i_above)
            I_below = IdentityGate(num_of_i_below)

            gates[i] = I_above % not_gate % I_below
