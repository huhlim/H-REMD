#!/usr/bin/env python

import os
from mpi4py import MPI

MPI_COMM = MPI.COMM_WORLD
MPI_SIZE = MPI_COMM.Get_size()
MPI_RANK = MPI_COMM.Get_rank()
MPI_KING = 0

def distribute_trajectory(n_replica):
    if MPI_RANK == MPI_KING:
        # get GPU indices
        if 'gpu_id' in os.environ:
            GPU_s = os.environ['gpu_id'].split(",")
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            GPU_s = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
        else:
            GPU_s = ['0']
        n_GPU = len(GPU_s)
        #
        traj_index = [[] for i in range(MPI_SIZE)]
        for i in range(n_replica):
            traj_index[i%MPI_SIZE].append((i, GPU_s[i%n_GPU]))  # replicaID, gpu_id
    else:
        traj_index = {}
    #
    traj_index = MPI_COMM.bcast(traj_index, root=MPI_KING)[MPI_RANK]
    #
    return traj_index

