#!/usr/bin/env python

import os
import sys
import numpy as np

from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *

from libcustom import construct_custom_restraint, read_custom_restraint
from libmpi import MPI_KING, MPI_SIZE, MPI_RANK, MPI_COMM, distribute_trajectory

DEBUG=True

class StateParameter:
    ''' This class can generate objects with different
        - force field
        - restraitns
        - simulation temperature
    '''
    def __init__(self, toppar, dt=0.002, T=298.0):
        self.ff = CharmmParameterSet(*toppar)
        self.dyntemp = T * kelvin
        #
        if not DEBUG:
            self.dynoutfrq = 25000  # 10 ps
        else:
            self.dynoutfrq = 2500  # 10 ps
        self.dyntstep = dt * picosecond
        self.langfbeta = 0.01 / picosecond
        #
        # system
        self.nonbondedMethod = PME
        self.switchDistance = 0.8 * nanometers
        self.nonbondedCutoff = 1.0 * nanometers
        self.constraints = HBonds
        #
    def set_psf(self, psf_fn):
        self.psf = CharmmPsfFile(psf_fn)
    def set_box(self, boxsize):
        self.psf.setBox(*boxsize)

    @classmethod
    def define_state(cls, psf_fn, boxsize, fn=None, \
                            toppar=[], dt=0.002, T=298.0, cons=[], custom=[]):
        dyntemp = T
        with open(fn) as fp:
            for line in fp:
                if line.startswith("#"): continue
                x = line.strip().split()
                if x[0] == 'toppar':
                    toppar.extend(x[1:])
                elif x[0] == 'dyntemp':
                    dyntemp = float(x[1])
                elif x[0] == 'dyntstep' or x[0] == 'dt' or x[0] == 'time_step':
                    dt = float(x[1])
                elif x[0] == 'cons':
                    # ref_fn, force_const, flat_bottom
                    cons = [x[1]] + [float(xi) for xi in x[2:]]
                    if len(cons) == 2: cons.append(0.0)
                elif x[0] == 'custom':
                    custom = [x[1], read_custom_restraint(x[2])]

        state = cls(toppar, dt=dt, T=dyntemp)
        state.set_psf(psf_fn)
        state.set_box(boxsize)
        state.create_system()
        state.create_integrator()
        if len(cons) != 0:
            state.construct_restraint(cons[0], force_const=cons[1], flat_bottom=cons[2])
        if len(custom) != 0:
            state.construct_custom_restraint(custom[0], custom[1])
        return state

    def create_system(self):
        system = self.psf.createSystem(self.ff, \
                                       nonbondedMethod = self.nonbondedMethod, \
                                       switchDistance = self.switchDistance, \
                                       nonbondedCutoff = self.nonbondedCutoff, \
                                       constraints = self.constraints)
        self.system = system

    def create_integrator(self):
        integrator =  LangevinIntegrator(self.dyntemp, \
                                         self.langfbeta, \
                                         self.dyntstep)
        self.integrator = integrator

    def construct_restraint(self, ref_fn, force_const=0.0, flat_bottom=0.0):
        if force_const == 0.0:
            return
        #
        if flat_bottom == 0.0:
            restr = CustomExternalForce("k0*dsq ; dsq=((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        else:
            restr = CustomExternalForce("k0*(max(d-d0, 0.0))^2 ; d=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            restr.addGlobalParameter("d0", flat_bottom*angstroms)
        restr.addPerParticleParameter("x0")
        restr.addPerParticleParameter("y0")
        restr.addPerParticleParameter("z0")
        restr.addPerParticleParameter('k0')
        #
        ref = PDBFile(ref_fn)
        atom_s = [(int(atom.residue.id), atom.name, atom.element.mass) \
                                            for atom in ref.topology.atoms()]
        calphaIndex = []
        for i,atom in enumerate(self.psf.topology.atoms()):
            if atom.name == 'CA':
                calphaIndex.append(i)
        #
        i_atm = -1
        for i,atom_crd in enumerate(ref.positions):
            if atom_s[i][1] != 'CA': continue
            #
            i_atm += 1
            mass = atom_s[i][2]
            k = force_const*mass*kilocalories_per_mole/angstroms**2
            par = list(atom_crd.value_in_unit(nanometers))
            par.append(force_const)
            restr.addParticle(calphaIndex[i_atm], par)
        self.system.addForce(restr)

    def construct_custom_restraint(self, ref_fn, custom_s):
        for custom in construct_custom_restraint(self.psf, ref_fn, custom_s):
            self.system.addForce(custom)

    def initialize_simulation(self, gpu_id='0'):
        platform = Platform.getPlatformByName("CUDA")
        properties = {}
        properties['CudaDeviceIndex'] = gpu_id
        #
        simulation = Simulation(self.psf.topology, \
                                self.system, \
                                self.integrator, \
                                platform, properties)
        return simulation

class Replica:
    def __init__(self, replica_id, prefix, prefix_prev=None):
        self.replica_id = replica_id
        self.prefix = prefix
        self.prefix_prev = prefix_prev
        #
    def set_replica(self, state):
        self.state = state
    def set_simulation(self, init, gpu_id='0'):
        self.simulation = self.state.initialize_simulation(gpu_id=gpu_id)
        self.simulation.context.setPositions(init)
        #
        if self.restart is not None:
            with open(self.restart, 'rb') as fp:
                self.simulation.context.loadCheckpoint(fp.read())
        #
        self.simulation.context.setVelocitiesToTemperature(self.state.dyntemp)
        self.simulation.reporters.append(\
                DCDReporter(self.trajout, self.state.dynoutfrq))

    def set_total_steps(self, dynsteps):
        self.simulation.reporters.append(
                PDBReporter(self.finalpdb, dynsteps))
        self.simulation.reporters.append(\
                StateDataReporter(self.logfile, self.state.dynoutfrq,\
                step=True, time=True, kineticEnergy=True, potentialEnergy=True,\
                temperature=True, progress=True, remainingTime=True, speed=True,\
                totalSteps=dynsteps, separator='\t'))

    def create_checkpoint(self):
        with open(self.restout, 'wb') as fout:
            fout.write(self.simulation.context.createCheckpoint())

    @ property
    def home(self):
        return 'replica.%d'%self.replica_id
    @ property
    def logfile(self):
        return '%s/%s.log'%(self.home, self.prefix)
    @ property
    def restout(self):
        return '%s/%s.restart'%(self.home, self.prefix)
    @ property
    def trajout(self):
        return '%s/%s.dcd'%(self.home, self.prefix)
    @ property
    def finalpdb(self):
        return '%s/%s.pdb'%(self.home, self.prefix)
    @ property
    def restart(self):
        if self.prefix_prev is not None:
            return '%s/%s.restart'%(self.home, self.prefix_prev)
        else:
            return None

class ReplicaExchange:
    def __init__(self, prefix=None, prefix_prev=None):
        self.prefix = prefix
        self.prefix_prev = prefix_prev
        #
        self.replica_s = []
        if not DEBUG:
            self.replica_exchange_rate = 5000 # 10 ps
        else:
            self.replica_exchange_rate = 500 # 1 ps
        self.traj_id_s = []

    @property
    def logfile(self):
        return '%s.log'%self.prefix
    @property
    def history_fn(self):
        return '%s.history'%self.prefix

    def initialize(self, init_str, psf_fn, boxsize, state_fn_s):
        if init_str.endswith("crd"):
            init = CharmmCrdFile(init_str)
        elif init_str.endswith("pdb"):
            init = PDBFile(init_str)
        #
        boxsize = np.array(boxsize) / 10.0  # boxsize in nm
        for replica_id, gpu_id in distribute_trajectory(len(state_fn_s)):
            replica = Replica(replica_id, self.prefix, prefix_prev=self.prefix_prev)
            if not os.path.exists(replica.home):
                os.mkdir(replica.home)
            #
            state = StateParameter.define_state(psf_fn, boxsize,\
                                                    fn=state_fn_s[replica_id])
            replica.set_replica(state)
            replica.set_simulation(init.positions, gpu_id=gpu_id)
            #
            self.replica_s.append(replica)
        #
        if MPI_RANK == MPI_KING:
            self.traj_id_s = range(len(state_fn_s))
            self.history = open(self.history_fn, 'wt', buffering=0)
            self.history.write("%5d  "%(0) + ' '.join(['%2d'%traj_id for traj_id in self.traj_id_s])+'\n')
            self.log = open(self.logfile, 'wt', buffering=0)

            
    def finalize(self):
        for replica in self.replica_s:
            replica.create_checkpoint()
        #
        if MPI_RANK == MPI_KING:
            self.log.close()
            self.history.close()

    def run(self, n_step):
        self.n_step = n_step
        if self.n_step%self.replica_exchange_rate != 0:
            sys.exit("Error: n_step % replica_exchange_rate != 0, where replica_exchange_rate = %d\n"%self.replica_exchange_rate)
        self.n_iter = int(self.n_step/self.replica_exchange_rate)
        #
        for replica in self.replica_s:
            replica.set_total_steps(self.n_step)

        if MPI_RANK == MPI_KING:
            accepted = 0
        #
        # start simulation
        for i_iter in range(self.n_iter):
            # run for a replica_exchange_rate steps
            replica_data = []
            for replica in self.replica_s:
                replica.simulation.step(self.replica_exchange_rate)
                replica.create_checkpoint()
                #
                energy = replica.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilocalories_per_mole)
                replica_data.append((replica.replica_id, replica.restout, energy, replica.state.dyntemp))
            #
            MPI_COMM.barrier()  # to synchronize
            before_swap = MPI_COMM.gather(replica_data, root=MPI_KING)
            #
            if MPI_RANK == MPI_KING:
                replica_index = {}
                for rank in range(len(before_swap)):
                    for index in range(len(before_swap[rank])):
                        replica_index[before_swap[rank][index][0]] = (rank, index)
                #
                # select two replica indices to swap
                #swap_i, swap_j = np.random.choice(np.arange(len(replica_index)), 2, replace=False)
                swap_i = np.random.randint(len(replica_index)-1) ; swap_j = swap_i+1
                ii = replica_index[swap_i]  # rank, index
                jj = replica_index[swap_j]  # rank, index
                #
                energy_ii = before_swap[ii[0]][ii[1]][2]
                energy_jj = before_swap[jj[0]][jj[1]][2]
                #
                chk_fn_s = []
                for rank in range(len(before_swap)):
                    chk_fn_s.append([data[1] for data in before_swap[rank]])
                chk_fn_s[ii[0]][ii[1]], chk_fn_s[jj[0]][jj[1]] = chk_fn_s[jj[0]][jj[1]], chk_fn_s[ii[0]][ii[1]]
            else:
                chk_fn_s = None
            #
            # temporarily swapping
            swapped_chk_fn_s = MPI_COMM.scatter(chk_fn_s, root=MPI_KING)
            #
            replica_data = []
            for i,replica in enumerate(self.replica_s):
                with open(swapped_chk_fn_s[i], 'rb') as fp:
                    replica.simulation.context.loadCheckpoint(fp.read())
                energy = replica.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilocalories_per_mole)
                #
                replica_data.append((replica.replica_id, swapped_chk_fn_s[i], energy, replica.state.dyntemp))
            #
            MPI_COMM.barrier()  # to synchronize
            after_swap = MPI_COMM.gather(replica_data, root=MPI_KING)
            #
            if MPI_RANK == MPI_KING:
                energy_ij = after_swap[ii[0]][ii[1]][2]
                energy_ji = after_swap[jj[0]][jj[1]][2]
                #
                temperature_i = after_swap[ii[0]][ii[1]][3]
                temperature_j = after_swap[jj[0]][jj[1]][3]
                #
                kT_i = (MOLAR_GAS_CONSTANT_R * temperature_i).value_in_unit(kilocalories_per_mole)
                kT_j = (MOLAR_GAS_CONSTANT_R * temperature_j).value_in_unit(kilocalories_per_mole)
                #
                delta = (energy_ij-energy_ii)/kT_i + (energy_ji-energy_jj)/kT_j
                crit = min(1.0, np.exp(-delta))
                #
                wrt = []
                wrt.append("STEP %5d : %2d %2d"%(i_iter+1, swap_i, swap_j))
                if np.random.random() < crit:   # swap accepted
                    accepted += 1
                    wrt.append("ACCEPTED")
                    self.traj_id_s[swap_i], self.traj_id_s[swap_j] = self.traj_id_s[swap_j], self.traj_id_s[swap_i]
                else:                           # swap rejected
                    wrt.append("REJECTED")
                    chk_fn_s[ii[0]][ii[1]], chk_fn_s[jj[0]][jj[1]] = chk_fn_s[jj[0]][jj[1]], chk_fn_s[ii[0]][ii[1]]
                wrt.append("ACCEPTANCE %6.2f"%(float(accepted)/(i_iter+1)*100.0))
                wrt.append("PROB  %6.2f"%(crit*100.0))
                wrt.append("TEMP  %6.1f %6.1f"%(temperature_i.value_in_unit(kelvin), temperature_j.value_in_unit(kelvin)))
                wrt.append("ENERGY  %12.3f %12.3f <-> %12.3f %12.3f"%(energy_ii, energy_jj, energy_ij, energy_ji))
                self.log.write(' '.join(wrt)+"\n")
                self.history.write("%5d  "%(i_iter+1) + ' '.join(['%2d'%traj_id for traj_id in self.traj_id_s])+'\n')
                sys.stdout.write(' '.join(wrt)+"\n")
            else:
                chk_fn_s = None
            #
            swapped_chk_fn_s = MPI_COMM.scatter(chk_fn_s, root=MPI_KING)
            #
            for i,replica in enumerate(self.replica_s):
                with open(swapped_chk_fn_s[i], 'rb') as fp:
                    replica.simulation.context.loadCheckpoint(fp.read())
        #
        MPI_COMM.barrier()

