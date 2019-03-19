# Quantum Computing Project
### Group 1 ###


A robust simulation of an n-qubit quantum computer was constructed using a network of tensor registers and operators,
with functionality up to 15 qubits on a standard classical computer system. This simulator was
implemented in Python 3.6.

## Dependencies
The package requires the following python packages to work properly

* _Numpy_
* _Scipy_
* _Matplotlib_
* _Sys_
* _Math_
* _Cmath_

## Installation Instructions
Download the repository in a folder of your choice and unzip the files. Then cd into the folder and run:

`
pip install .
`

This will install the package along with all its dependencies.

## Structure of repository

All simulation source code is located in the _src_ folder. Within this file there is a _Sparse Matrix_ file which
contains the structure with which _operators_ and _registers_ are based on.

HTML documentation for all _.py_ files are located in the _docs_ folder.

Modules from within the _src_ folder can be accessed by files out with this folder using the import commands:

`import src.FILENAME`

or

`from src.FILENAME import MODULENAME`

![Alt text](Images/Flowchart.png?raw=true "Title")

## Run example programmes

Example programmes which simulate the effects of Grover's and Shor's algorithms can be run from the main directory using
the commands:

`$ python grovers_algorithm.py n_qubits [1:'multiples-of' or 2:'exponents-of'] target-number optional(number of target
 states)`

`$ python shors_algorithm.py n_qubits test_value`

## Authors

* **Freddie Ferguson**
* **Iain Mclaughaln**
* **Harsh Rathee**
* **Ola Sobieska**
