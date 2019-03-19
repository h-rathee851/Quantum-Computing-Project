n = 100
x_prime = 5
xs = range(2**n)


def f(x):
    if x == x_prime:
        return 1
    else:
        return 0


def classic_grover():
    for x in xs:
        if f(x):
            return x


print(classic_grover())

