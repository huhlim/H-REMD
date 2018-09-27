mpirun -np 2 ../rex.py test.equil.pdb --state state_0 state_1 --psf test.psf --box $(cat boxsize) --n_step 5000000 --prefix r0.0
mpirun -np 2 ../rex.py test.equil.pdb --state state_0 state_1 --psf test.psf --box $(cat boxsize) --n_step 5000000 --prefix r0.1 --restart r0.0
mpirun -np 2 ../rex.py test.equil.pdb --state state_0 state_1 --psf test.psf --box $(cat boxsize) --n_step 5000000 --prefix r0.2 --restart r0.1
mpirun -np 2 ../rex.py test.equil.pdb --state state_0 state_1 --psf test.psf --box $(cat boxsize) --n_step 5000000 --prefix r0.3 --restart r0.2
mpirun -np 2 ../rex.py test.equil.pdb --state state_0 state_1 --psf test.psf --box $(cat boxsize) --n_step 5000000 --prefix r0.4 --restart r0.3
