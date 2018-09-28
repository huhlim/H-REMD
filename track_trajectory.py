#!/usr/bin/env python

import os
import sys
import numpy as np
import mdtraj
import argparse

def read_history(prefix_s):
    history = []
    for prefix in prefix_s:
        history_fn = '%s.history'%prefix
        if not os.path.exists(history_fn):
            sys.exit("Error: there is no such file, %s\n"%history_fn)
        #
        with open(history_fn) as fp:
            h = [line.strip().split()[1:] for line in fp.readlines()]
        history.extend(h[:-1])
    history = np.array(history, dtype=int)
    return history

def main():
    arg = argparse.ArgumentParser(prog='rex.track_traj')
    arg.add_argument('-t', '--top', dest='top_fn', required=True)
    arg.add_argument('-p', '--prefix', dest='prefix_s', required=True, nargs='+')
    arg.add_argument('-o', '--output', dest='output_prefix', default='traj')
    arg.add_argument('-s', '--skip', dest='skip', default=5)
    #
    if len(sys.argv) == 1:
        arg.print_help()
        return
    arg = arg.parse_args()

    history = read_history(arg.prefix_s)
    history = history[arg.skip-1::arg.skip].T

    top=mdtraj.load(arg.top_fn)
    #
    n_replica = history.shape[0]
    n_frame = history.shape[1]
    #
    replica = []
    for k in range(n_replica):
        for prefix in arg.prefix_s:
            replica_fn = 'replica.%d/%s.dcd'%(k, prefix)
            replica.append(mdtraj.load(replica_fn, top=top))
    replica = mdtraj.join(replica, check_topology=False)
    #
    for k in range(n_replica):
        frame = np.array([n_frame*j+i for i,j in enumerate(history[k])])
        traj = replica.slice(frame)
        traj.save("%s.%d.dcd"%(arg.output_prefix, k))

if __name__ == '__main__':
    main()


