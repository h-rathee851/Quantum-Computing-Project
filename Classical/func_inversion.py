"""
Possible application of Grover's Algorithm as a function inversion method.
This tests various inputs through a hard to invert algorithm in this case
a hash function (md5) and through testing various 'passwords' by hashing them
and comparing to a stored hash solution to find the original password. This
method is similar to a dictionary attack.
Inspired from 'daytonellwagner' on YouTube.
"""

import hashlib

n = 5
xs = ['password', '1234', 'qwertyuiop', 'dog']
x_prime = '81dc9bdb52d04dc20036dbd8313ed055'


def g(x):
    m = hashlib.md5()
    m.update(x.encode('utf-8'))
    return m.hexdigest()


def f(x):
    y_prime = g(x)
    if y_prime == x_prime:
        return 1
    else:
        return 0


def classic_grover():
    for x in xs:
        if f(x):
            return x


print(classic_grover())


"""
to get hashed password:
$ m = hashlib.md5()
$ m.update('1234'.encode('utf-8))
$ m.hexdigest()
"""