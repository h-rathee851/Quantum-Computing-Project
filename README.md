# Quantum-Computing-Project

Overleaf edit link: https://v2.overleaf.com/6298525371rxmqzpwqrcrk 


Useful youtube series going over the same info as in the lectures: https://www.youtube.com/watch?v=TAzZKAdX2Tw&list=PLIxlJjN2V90w3KBWpELOE7jNQMICxoRwc

YT lecture about QP: https://www.youtube.com/watch?v=F_Riqjdh2oM&t=3465s&fbclid=IwAR3-faIV9sNRyjs1Hptwua5mqZl1pv5NGePMjHtj0GIM_gX5CPrFMVatfyY

Book about quantum computing by Nielsen: https://drive.google.com/file/d/1xmP3ZcVaypLzFtWMtpJhPuT6D3t__rC_/view?fbclid=IwAR0QROcKIVLmSHhGAQ5O0bIqK-oeysW3yGSfiL-Vf81yuBrmRBr0gSyY6Zc

Gates to be used:

Hadamard
Phase Shift
c-Not

Pauli-X
Pauli-Y
Pauli-Z

C-V
Swap
C2-Not
c-Vdagger
Toffoli
Quantum Adder


Example documentation:

def c_V_gate(control_qubit, target_qubit):
    """
    :param control_qubit: (qubit) A qqubi
    :param target_qubit:
    :return: (np.array) a ma
    """
    matrix = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1j]])
    input = np.concatenate((control_qubit, target_qubit), axis=None)
    out = np.dot(matrix, input)
    print(out)
    return out
