# Obtain dynamical tides in an ocean with varying stratification (B.V. frequency).
#
# Guideline:
# https://computationalmechanics.in/parallelizing-for-loop-in-python-with-mpi/
#
# Guidelines on Reduce: 
# https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/
#
# Simple run: 
# mpirun -np 6 python3 titan_Hvar_mpi.py
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


#N2_size = 100
#N2_array = np.linspace(1e-8, 1e-6, N2_size)

G   = 6.67e-8
label = 'tau9_Hvar_mpi'
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
N2      = 1e-8         # Ocean stratification
tau     = 1e9       # frictional dissipation

# Chebyshev solver
N = 80         # number of Chebyshev polynomials
Lmax = 80
M = 2

# Initial calculations
L       = np.arange(abs(M), Lmax+1)
Rp      = R             # body radius
b       = np.pi
sma     = (G*(Ms+ms)*(Ts*3600)**2/4/np.pi**2)**(1/3)     # satisfy Kepler's third law
oms     = 2*np.pi/Ts/3600           # orbital frequency
Om      = oms                # rotational frequency in synchronous rotation
ome     = oms     # eccentricity tidal frequency

# Varying depth calculations
H_spacing   = 0.05 # in km
H_spacing   = 1 # in km
H_vec       = np.arange(200, 400 + H_spacing, H_spacing)*1e5
H_size      = len(H_vec)
eta_vec     = (R-H_vec)/R
Rc_vec      = R*eta_vec
a_vec       = np.pi*eta_vec
rhoc_vec    = 3*ms/(4*np.pi*Rc_vec**3) - rhow*((R/Rc_vec)**3 - 1)
k2_vec = np.zeros(len(H_vec))
E_vec = np.zeros(len(H_vec))

# Main calculation
def main_calc():
    
    cheb = cheby(npoints=N, loend=a_vec[ind], upend=b)

    dyn = dynamical(cheb, ome, Om, ms, Ms, sma, Rp,
                 rho=rhow, rhoc=rhoc_vec[ind], Rc=Rc_vec[ind],
                 l=L, m=M, tau=tau, x1=a_vec[ind], x2=b,
                 tides='ee', e=e, N2=N2)
    
    dyn.solve(kind='bvp-core')

    return dyn.k[0], sum(dyn.E)

# MPI 
num_per_rank = H_size//size

H_lower_bound = rank*num_per_rank 
H_upper_bound = (rank + 1)*num_per_rank

print("Processor ", rank, ": iterations ", H_lower_bound," to ", H_upper_bound - 1, flush=True)

comm.Barrier() # start parallel processes

start_time = time.time()

for ind in range(H_lower_bound, H_upper_bound):

    k2_vec[ind], E_vec[ind] = main_calc() 
    
    print("Iteration ", ind, " done in processor ", rank)

if rank == 0:
    k2_global = np.zeros(len(H_vec))
    E_global = np.zeros(len(H_vec))

else:
    k2_global = None
    E_global = None

comm.Barrier() # end prallel processes

comm.Reduce( k2_vec , k2_global, op=MPI.SUM, root=0)

comm.Reduce( E_vec , E_global, op=MPI.SUM, root=0)

stop_time = time.time()

if rank ==0:
    # Complete processes that did not evenly fit the number of processors
    for ind in range(size*num_per_rank, H_size):

        k2_global[ind], E_global[ind] = main_calc()

    print("time spent with ", size, " threads in minutes")
    print("-----", int((stop_time - start_time)/60 ), "-----")

    dill.dump([H_vec, k2_global, E_global], file=open('./k2Q_Rcvec_L{}_N{}{}.pickle'.format(np.max(L), N, label), 'wb') )
