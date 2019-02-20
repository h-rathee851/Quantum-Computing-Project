import numpy as np
from numpy.linalg import norm
import cmath
import matplotlib.pyplot as plt

class SparseMatrix:
    # rows and columns give the dimensions of the matrix
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.matrix = {}
        return

    def setElement(self, i, j, m):
        if i >= self.rows or j >= self.columns:
            raise IndexError('Index out of range')
        if abs(m) != 0:
            self.matrix[(i,j)] = m
        elif abs(m) == 0 and (i,j) in self.matrix:
            del self.matrix[(i,j)]

    def getElement(self,i,j):
        if (i,j) in self.matrix:
            return self.matrix[(i,j)]
        else:
            return 0

    def getHermTranspose(self):
        # TODO:
        return

    def __str__(self):
        rep = ''
        for i in range(0,self.rows):
            row = ''
            for j in range(0,self.columns):
                row += str(self.getElement(i,j))+'\t'
            rep += row + '\n'
        return rep

    # Define inner product
    def innerProduct(self, other):
        result = SparseMatrix(self.rows,other.columns)
        if self.columns == other.rows:
            for (i,j) in self.matrix:
                for (k,l) in other.matrix:
                    if (j == k):
                        val = result.getElement(i,l)
                        result.setElement(i, l, val + (self.matrix[(i,j)]
                                                        * other.matrix[(k,l)]))
        else:
            raise ValueError('Incompatible Dimentions')
        return result

    #Define outer product
    #Other on the right
    def outerProduct(self, other):
        result = SparseMatrix(self.rows*other.rows,self.columns*other.columns)
        for (i,j) in self.matrix:
            for (k,l) in other.matrix:
                result.setElement( ((i*other.rows)+k), ((j*other.columns)+l),
                                    self.matrix[(i,j)]*other.matrix[(k,l)] )
        return result

    @staticmethod
    def add(m1,m2):
        return


def test():
    sm = SparseMatrix(2, 2)
    sm2 = SparseMatrix(2, 2)
    sm.setElement(0, 0, 5)
    sm2.setElement(0, 0, 20)
    print(sm)
    print(sm.getElement(0, 0))
    print(sm.matrix)
    sm3 = sm.innerProduct(sm2)
    print(sm3)
    sm4 = sm3.outerProduct(sm)
    print(sm4)

# test()
