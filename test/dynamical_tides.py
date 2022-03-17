# Solve dynamical tides in an index-one polytrope.
#
# Ben Idini, May 2020.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn as jn
from interiorize.solvers import cheby
from interiorize.polytrope import dynamical, gravity
from matplotlib import rcParams
import pdb

rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
rcParams['mathtext.fontset'] = 'cm'

# Specify parameters
N = 200                                # number of Chebyshev polynomialsi
a = 1e-10*np.pi
b = np.pi*1.                              # planet's surface   
M = 2
L = np.arange(M if M>1 else 3,M+30,2)
#L = np.arange(M if M>0 else 3,M+150,2)
# note: k11 is inf as the gravitational pull is zero

rhoc = 3.8                              # central density
rhoc = 4.38                              # central density
G   = 6.67e-8                           # gravitational universal constant
Ts = 15.945*24 # Titan
Ts = 85.2 # Europa
Ts = 171.7 # Ganymede
Ts = 400.5 # Callisto
Ts = 42.46 # Io
Om  = 2*np.pi/(10.6*3600)                # Saturn spin rate
Om  = 2*np.pi/(9.9*3600)                # Jupiter spin rate
oms = 2*np.pi/Ts/3600
tau = 1e10
#tau = 1e30

M = -M # convention sets retrograde tides with negative m
om  = M*(oms - Om)     # tidal frequency
print('tidal frequency {} mHz'.format(om*1e3))

omd = np.sqrt(4*np.pi*G*rhoc)            # dynamical frequency
ms = 1.345e26 # Titan mass
ms = 4.8e25         # Europa mass
ms = 1.48e26 # Ganymede mass
ms = 1.076e26  # Callisto mass
ms  = 8.931e25                          # Io mass
Mj = 1.898e30
Ms = 5.68e29
Rj = 6.99e9
Rs = 5.82e9
sma   = (G*(Ms+ms)*(Ts*3600)**2/4/np.pi**2)**(1/3)                         # Io semi-major axis
sma   = (G*(Mj+ms)*(Ts*3600)**2/4/np.pi**2)**(1/3)                         # Io semi-major axis
Rp = Rj
K = 2.1e12
k = np.sqrt(2*np.pi*G/K)

## Solve the problem
cheb = cheby(npoints=N, loend=a, upend=b)
dyn = dynamical(cheb,om,Om, Mj, ms , sma, Rp,l=L, m=M,tau=tau,x0=a,xr=b)
dyn.solve(kind='bvp')
#dyn.solve(kind='bvp-core')

# Plot spectral coefficients.
# gravity at the surface is the sum of the spectral coefficients. If they reach a round-off plateau, we are adding crap to the solution.
plt.figure(figsize=(4,4))
[plt.semilogy( np.arange(1,N+1),abs(np.split(dyn.an,len(L))[i]),label=r'$\ell,m = {},-2$'.format(L[i]),linewidth=1,alpha=0.6) for i in range(0,2)] 
plt.xlim((0,N))
plt.minorticks_on()
plt.tick_params(which='both',direction='in',top=True,right=True)
for tick in plt.xticks()[1]+plt.yticks()[1]:
    tick.set_fontname("DejaVu Serif")
plt.xlabel(('$n$'))
plt.ylabel(('Flow potential ($\psi$) Chebyshev coefficients, $|a_n|$'))
plt.legend()
plt.tight_layout()
plt.show()

xi = cheb.xi
plt.subplot(311)
plt.plot(xi, dyn.psi[0],label='$\psi$')
plt.legend()
plt.subplot(312)
plt.plot(xi, dyn.dpsi[0],label='$\partial \psi/\partial x$')
plt.legend()
plt.subplot(313)
plt.plot(xi, dyn.d2psi[0],label='$\partial^2 \psi/\partial x^2$')
plt.legend()
plt.show()

# Plot radial functions of the potential
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(xi/b, dyn.psi[0]/max(abs(dyn.psi[0])), label=r'$\ell,m = 2,-2$')
plt.plot(xi/b, dyn.psi[1]/max(abs(dyn.psi[1])), '--',label=r'$\ell,m = 4,-2$')
plt.plot(xi/b, (xi/b)**2,'-k',linewidth=1,label=r'$r^2$')
plt.plot(xi/b, (xi/b)**4,'--k',linewidth=1,label=r'$r^4$')
plt.legend()
plt.xlabel((r'Normalized radius, $r/R_p$'))
plt.ylabel((r'Potential $\psi_{\ell,m}$'))
plt.xlim((0,1))
plt.minorticks_on()
plt.tick_params(which='both',direction='in',top=True,right=True)
for tick in plt.xticks()[1]+plt.yticks()[1]:
    tick.set_fontname("DejaVu Serif")
plt.tight_layout()

plt.subplot(122)
plt.plot(xi/b, dyn.phi_dyn[0]/max(abs(dyn.phi_dyn[0])), label=r'$\ell,m = 2,-2$')
plt.plot(xi/b, dyn.phi_dyn[1]/max(abs(dyn.phi_dyn[1])), '--',label=r'$\ell,m = 4,-2$')
plt.legend()
plt.xlabel((r'Normalized radius, $r/R_p$'))
plt.ylabel((r'Dynamic gravitational potential, $\phi^{dyn}_{\ell,m}$'))
plt.xlim((0,1))
plt.minorticks_on()
plt.tick_params(which='both',direction='in',top=True,right=True)
for tick in plt.xticks()[1]+plt.yticks()[1]:
    tick.set_fontname("DejaVu Serif")
plt.tight_layout()
plt.show()

# Print the percentile dynamic correction to the love number
[print('l = {}: {:.2f}%'.format(L[i], dyn.dk[i] )) for i in range(0,3)]

print('log(Q_2): {}'.format(np.log10(abs( dyn.Q[0] ))))
