"""
08/03/2018
Andreas Malekos - Quantum Computing Project

File that runs Grover's algorithm in parallel and returns the most likely
state that is tagged.

For now, this can only work in single target mode.
"""

from qc_simulator.qc import *
from qc_simulator.grover import grover
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_grover_parallel(oracle, n_runs):
    """
    Function that runs grover many times in parallel

    Inputs:
        oracle: oracle gate
        n_runs: number of runs
    Outputs:
        results: list of results
    """
    pass




if __name__=='__main__':

    n_runs = 10

    if rank == 0:
        print('test')
        results_total = np.zeros(n_runs)
    else:
        results_total = None

    results_local = np.zeros( int(n_runs/size))

    comm.Scatterv(results_total, results_local, root=0)

    print('Rank {} has data: {}'.format(rank, results_local))
    results_local = results_local + rank + 1
    print('From process {}: {} '.format(rank, results_local))

    comm.Gatherv(results_local, results_total, root=0)
    print('From proces {}: Total data is: {}'.format(rank, results_total))
