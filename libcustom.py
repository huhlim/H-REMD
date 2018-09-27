#!/usr/bin/env python

from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *

class Custom:
    def __init__(self, r_type, n_atom, n_param, i_atm, prm):
        self.r_type = r_type
        self.n_atom = n_atom
        self.n_param = n_param
        self.i_atm = i_atm
        self.prm = prm

def construct_custom_restraint(psf, ref, custom_s):
    rsr = []
    if len(custom_s) == 0:
        return rsr
    #
    if ref.endswith(".pdb"):
        pdb = PDBFile(ref)
    else:
        pdb = CharmmCrdFile(ref)
    crd = pdb.positions
    mass = [atom.element.mass for atom in pdb.topology.atoms()]
    #
    bond = CustomBondForce("k * (r-r0)^2")
    bond.addPerBondParameter('k')
    bond.addPerBondParameter('r0')
    #
    pos = CustomExternalForce("k0*dsq ; dsq=((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    pos.addPerParticleParameter("x0")
    pos.addPerParticleParameter("y0")
    pos.addPerParticleParameter("z0")
    pos.addPerParticleParameter('k0')
    #
    for custom in custom_s:
        if custom.r_type == 'bond':
            p = (custom.prm[0]*kilocalories_per_mole/angstroms**2,\
                 custom.prm[1]*angstroms)
            bond.addBond(custom.i_atm[0], custom.i_atm[1], p)
        elif custom.r_type == 'angle':
            pass
        elif custom.r_type == 'torsion':
            pass
        elif custom.r_type == 'position':
            i_atm = custom.i_atm[0]
            p = list(crd[i_atm].value_in_unit(nanometers))
            p.append(custom.prm[0]*mass[i_atm]*kilocalories_per_mole/angstroms**2)
            pos.addParticle(i_atm, p)
    #
    if bond.getNumBonds() > 0:
        rsr.append(bond)
    if pos.getNumParticles() > 0:
        rsr.append(pos)
    return rsr

def read_custom_restraint(custom_file):
    custom_restraints = []
    if custom_file is None:
        return custom_restraints

    with open('%s'%custom_file) as fp:
        for line in fp:
            if line.startswith("#"):
                continue
            x = line.strip().split()
            r_type = x[0]
            n_atom = int(x[1])
            n_param = int(x[2])
            i_atm = [int(xi)-1 for xi in x[3:3+n_atom]]
            prm = [float(xi) for xi in x[3+n_atom:]]
            custom = Custom(r_type, n_atom, n_param, i_atm, prm)
            custom_restraints.append(custom)
    return custom_restraints

