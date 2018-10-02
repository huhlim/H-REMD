#!/usr/bin/env python

import os
import sys
import time
import argparse

from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *

from librex import ReplicaExchange
from libmpi import MPI_RANK, MPI_KING, MPI_COMM

def main():
    arg = argparse.ArgumentParser(prog='rex')
    arg.add_argument(dest='input_pdb')
    arg.add_argument('-p', '--prefix', dest='prefix', required=True)
    arg.add_argument('-r', '--restart', dest='prefix_prev', default=None)
    arg.add_argument('-s', '--state', dest='state_s', nargs='*', required=True)
    arg.add_argument('-n', '--n_step', dest='n_step', type=int)
    arg.add_argument('--psf', dest='psf_fn', nargs='+', required=True)
    arg.add_argument('--box', dest='boxsize', nargs=3, type=float, required=True)
    arg.add_argument('--exchange_rate', dest='replica_exchange_rate', type=int, default=None)

    #
    if len(sys.argv) == 1:
        arg.print_help()
        return
    arg = arg.parse_args()
    if len(arg.state_s) < 2:
        sys.exit("Number of states is less than 2!\n")
    if len(arg.psf_fn) > 1:
        arg.input_pdb = arg.psf_fn[1]
    #
    if MPI_RANK == MPI_KING:
        t_init = time.time()
    #
    runner = ReplicaExchange(prefix=arg.prefix, prefix_prev=arg.prefix_prev)
    if arg.replica_exchange_rate is not None:
        runner.replica_exchange_rate = arg.replica_exchange_rate
    runner.initialize(arg.input_pdb, arg.psf_fn[0], arg.boxsize, arg.state_s)
    runner.run(arg.n_step)
    #
    if MPI_RANK == MPI_KING:
        t_final = time.time()
        t_spend = t_final - t_init
        #
        runner.log.write("ELAPSED TIME:     %8.2f SECONDS\n"%t_spend)
    else:
        t_spend = 0.0

    t_spend = MPI_COMM.bcast(t_spend, root=MPI_KING)
    for replica in runner.replica_s:
        with open(replica.logfile, 'at') as fout:
            fout.write("ELAPSED TIME:     %8.2f SECONDS\n"%t_spend)

    runner.finalize()

if __name__=='__main__':
    main()
