# Defines a setup file which can be used with `pip install .` to install all
# packages required.

from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='Quantum Computer Simulation',
    version='1',
    description='Simulation of a Quantum Network using TensorNetwork Methods',
    url='https://github.com/IainMcl/Quantum-Computing-Project/tree/master/QCP-Group-1-2019',
    author='Iain Mclaughlan',
    author_email='s1524154@sms.ed.ac.uk',
    packages=['src'],
    install_requires=['matplotlib', 'numpy', 'scipy'])
