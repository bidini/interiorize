# Obtain dynamical tides in an ocean with varying forcing frequency in eccentricity tides.
#
# Guideline:
# https://computationalmechanics.in/parallelizing-for-loop-in-python-with-mpi/
#
# Guidelines on Reduce: 
# https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/
#
# Simple run: 
# mpirun -np 6 python3 titan_omvar_mpi.py
#
# 03/2023
# Ben Idini

import numpy as np
from mpi4py import MPI
import time
import pdb
import dill

from interiorize.solvers import cheby
from interiorize.hough import dynamical


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

G   = 6.67e-8
label = 'g6_tau9_mpi'
# Titan
R       = 2575e5    # Mean radius (cm)
rhom    = 1.8798     # Mean density (g/cc)
ms      = 1.3452e26    # mass (g)
Ts      = 15.945*24 # orbital period (hours)
e       = 0.0288     # Eccentricity

# Saturn
Ms = 5.683e29
Rs = 5.8232e9

# Model parameters
rhow    = 1.
tau     = 1e9       # frictional dissipation
H	= 300e5
N2	= 1.4454e-8

# Chebyshev solver
N = 80         # number of Chebyshev polynomials
Lmax = 80
M = 2

# Initial calculations
L       = np.arange(abs(M), Lmax+1)
Rp      = R             # body radius
eta	= (R-H)/R
Rc	= R*eta
a	= np.pi*eta
rhoc	= 3*ms/(4*np.pi*Rc**3) - rhow*((R/Rc)**3 - 1)
b       = np.pi

# Varying om calculations (orbital frequency)
oms_size   = 12500 
oms_vec      = np.linspace(1, 3.312, oms_size)*2*np.pi/Ts/3600
sma_vec	= (G*(Ms+ms)/oms_vec**2)**(1/3)     # satisfy Kepler's third law
Om_vec = oms_vec  # rotational frequency in synchronous rotation
ome_vec = oms_vec # eccentricity tidal frequency
k2_vec = np.zeros(len(oms_vec))
E_vec = np.zeros(len(oms_vec))

# Main calculation
def main_calc():
    
    cheb = cheby(npoints=N, loend=a, upend=b)

    dyn = dynamical(cheb, ome_vec[ind], Om_vec[ind], ms, Ms, sma_vec[ind], Rp,
                 rho=rhow, rhoc=rhoc, Rc=Rc,
                 l=L, m=M, tau=tau, x1=a, x2=b,
                 tides='ee', e=e, N2=N2)
    
    dyn.solve(kind='bvp-core')

    return dyn.k[0], sum(dyn.E)

# MPI 
num_per_rank = oms_size//size

oms_lower_bound = rank*num_per_rank 
oms_upper_bound = (rank + 1)*num_per_rank

print("Processor ", rank, ": iterations ", oms_lower_bound," to ", oms_upper_bound - 1, flush=True)

comm.Barrier() # start parallel processes

start_time = time.time()

for ind in range(oms_lower_bound, oms_upper_bound):

    k2_vec[ind], E_vec[ind] = main_calc() 
    
    print("Iteration ", ind, " done in processor ", rank)

if rank == 0:
    k2_global = np.zeros(len(oms_vec))
    E_global = np.zeros(len(oms_vec))

else:
    k2_global = None
    E_global = None

comm.Barrier() # end prallel processes

comm.Reduce( k2_vec , k2_global, op=MPI.SUM, root=0)

comm.Reduce( E_vec , E_global, op=MPI.SUM, root=0)

stop_time = time.time()

if rank ==0:
    # Complete processes that did not evenly fit the number of processors
    for ind in range(size*num_per_rank, oms_size):

        k2_global[ind], E_global[ind] = main_calc()

    print("time spent with ", size, " threads in minutes")
    print("-----", int((stop_time - start_time)/60 ), "-----")

    dill.dump([oms_vec, k2_global, E_global], file=open('./k2Q_omsvec_L{}_N{}{}_H{}.pickle'.format(np.max(L), N, label, int(H/1e5)), 'wb') )
