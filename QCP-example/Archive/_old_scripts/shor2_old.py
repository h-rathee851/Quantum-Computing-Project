"""
Some old internal functions from shor 2 old
"""


def SumGate(self, n=3):
    """
    unlikely that it can be applied to n =\=3
    but will keep just in case for now, only used as n = 3
    """
    I = IdentityGate()
    M = I%CUGate(not())
    M = CUGate(Not(),1,1,1)*M
    return M

def CarryGate(self, n = 4):
    """
    like sum, likely only usable with 4
    """
    I = IdentityGate()
    M = I%build_c_c_not()
    M = (I%CUGate(Not())%I)*M
    M = build_c_c_not(1,0)*M #CQubit I CQubit TQubit
    return M
