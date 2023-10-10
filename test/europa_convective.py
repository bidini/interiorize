# Solve the dynamical tides in a Poincare problem (uniform density, full Coriolis). 
# The ocean is fully mixed and convective
#
# Ben Idini, Nov 2022.

import numpy as np
from scipy.special import spherical_jn as jn
from scipy.special import sph_harm as SH

from interiorize.solvers import cheby
from interiorize.poincare import dynamical
import interiorize.plottools as my_plots

from matplotlib import rcParams
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class
from matplotlib.ticker import MaxNLocator

import pdb
import dill


# PLOT DEFINITIONS
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
rcParams['mathtext.fontset'] = 'cm'

def my_plt_opt():

    plt.minorticks_on()
    plt.tick_params(which='both', direction='in', top=True, right=True)
    for tick in plt.xticks()[1] + plt.yticks()[1]:
        tick.set_fontname("DejaVu Serif")

    plt.tight_layout()

    return

## INPUT PARAMETERS
G   = 6.67e-8                           # gravitational universal constant

# Europa
R       = 1561e5    # Mean radius (cm)
rhom    = 3.013     # Mean density (g/cc)
MOI     = 0.346     # Moment of inertia
e       = 0.009     # Eccentricity
H       = 150e5     # Ocean thickness (cm)
rhow    = 1.       # Ocean density
ms      = 4.8e25    # Europa mass (g)
Ts      = 85.228344 # Europa's orbital period (hours)

# Jupiter
Mj = 1.898e30
Rj = 6.99e9
tau = 1e7

# Chebyshev solver
N = 200         # number of Chebyshev polynomialsi
Lmax = 200
M = 2
save = False
label = 'tau7'

L = np.arange(M if M>1 else 3, M+Lmax, 2)

# Prepare solver 
# eta = 0.9 # Europa
# eta = 0.87 # Titan
# List of candidate resonances
# 0.80152R
# 0.80606R
# 0.80677R
# 0.80848R
# 0.87093R
# 0.87677R
# 0.88182R
# 0.88737R
# 0.92253R

eta     = 0.9
Rc      = R*eta        # core radius
Rp      = R             # body radius
a       = Rc/Rp*np.pi
b       = np.pi*1.                              # planet's surface   
rhoc    = 3*ms/(4*np.pi*Rc**3) - rhow*((Rp/Rc)**3 -1)  # fit Europa's mass
sma     = (G*(Mj+ms)*(Ts*3600)**2/4/np.pi**2)**(1/3)     # satisfy Kepler's third law                   
oms     = 2*np.pi/Ts/3600           # orbital frequency
Om      = oms                # Europa's rotational frequency
om      = M*(oms - Om)     # conventional tidal frequency
ome     = oms     # eccentricity tidal frequency
ome = 0.9e-5 
Om = ome

print('conventional tidal frequency {} mHz'.format(om*1e3))
print('eccentricity tidal frequency {} mHz'.format(ome*1e3))

# Initialize the solver tool
cheb = cheby(npoints=N, loend=a, upend=b)

# Set the problem matrix
dyn = dynamical(cheb, ome, Om, ms, Mj, sma, Rp, 
                rho=rhow, rhoc=rhoc, Rc=Rc,
                l=L, m=M, tau=tau, x1=a, x2=b,
                tides='ee', e=e)


# Solve the linear problem
dyn.solve(kind='bvp-core') # flow with ocean bottom

# Save a copy of results
if save:
    dill.dump(dyn.an, file=open('/Users/benja/Documents/projects/europa/results/an_Rc{}p_L{}_N{}{}.pickle'.format(int(eta*100), np.max(L), N, label), 'wb') )

##################################################

# Print the percentile dynamic correction to the love number
[print('l = {}: {:.20f}%'.format(L[i], dyn.dk[i]*100 )) for i in range(0,3)]

print('log(Q_2): {}'.format(np.log10(abs( dyn.Q[0] ))))

print(dyn.k_hydro(2))

###################################################
# Sanity checks on displacement

# degree 2 displacement at the surface obtained from psi.
xi2 = dyn.flow[0][0]

# Hydrostatic degree-2 displacement at the surface obtained from the boundary condition.
xih2 = (dyn.k_hydro(2)+1)*dyn.Ulm(2,2)/dyn.g

###################################################

## PLOT RESULTS

# Chebyshev spectral coefficients.
# NOTE: gravity at the surface is the sum of the spectral coefficients. If they reach a round-off plateau, we have recovered the entire solution.

def my_plot1():
    plt.figure(figsize=(4,4))
    [ plt.semilogy( np.arange(1,N+1), abs(np.split(dyn.an, len(L))[i]), label=r'$\ell = {}$'.format(L[i]), linewidth=1, alpha=0.6) for i in range(0, len(L))] 
    plt.xlim((0,N))
    plt.xlabel(('$n$'))
    plt.ylabel(('$|a_n|$'))
    my_plt_opt()
    if save:
        plt.savefig('/Users/benja/Documents/projects/europa/results/an_Rc{}p_L{}_N{}{}.png'.format(int(eta*100), np.max(L), N, label), dpi=1200)
    else:
        plt.show()


# Radial displacement shells
def field(p, theta, varphi=0):
    """
    Get the total field of a spherical harmonic decomposition while summing over all degree. The result is a cross section at a given azimuthal angle.
    p: radial function.
    theta: colatitude.
    varphi: azimuthal angle
    """

    field_grd = np.zeros((len(theta), len(p[0])), dtype=complex)
    
    for i in np.arange(len(L)):
        p_grd, th_grd = np.meshgrid(p[i], theta)
        field_grd += p_grd*SH(M, L[i], varphi, th_grd)

    return field_grd, th_grd 

# generate grids for plotting.
def my_plot2():
    print('start contourf plot')
    x = cheb.xi/np.pi
    th = np.linspace(0, np.pi, 1000)
    disp_grd, th_grd = field(dyn.flow, th)
    x_grd = np.meshgrid(x, th)[0]
    disp_grd_m = np.real(disp_grd)/100
    vmin = np.min(disp_grd_m)
    vmax = np.max(disp_grd_m)

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    #cb = ax.contourf(th_grd, x_grd, abs(disp_grd_m), levels=500,vmin=0,vmax=30,  cmap='inferno')#'YlOrBr'
    log_disp = np.log10(abs(disp_grd_m))
    log_disp[log_disp<=0] = 0
    cb = ax.contourf(th_grd, x_grd, log_disp, levels=600, vmin=0,  cmap='inferno', extend='min')#'YlOrBr'
    #cb = ax.contourf(th_grd, x_grd, abs(disp_grd_m), levels=600, cmap='inferno')#'YlOrBr'
#    ax.contour(th_grd, x_grd, disp_grd_m, levels=10, colors='white', linewidths=1)#'YlOrBr'
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.axis("off")
    ax.set_theta_zero_location('N')
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_ylim([0,1])
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    colorb = fig.colorbar(cb, pad=-.1, ax=ax)
    colorb.set_label(r'$\log \xi_r$')
    #colorb.set_label(r'm')
    colorb.ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='upper'))
    my_plt_opt()
    '''
    inset = inset_axes(ax, width=3, height=3,
                       bbox_transform=ax.transAxes,
                       axes_class=get_projection_class('polar'),
                       bbox_to_anchor=(0.5,0.5),
                       loc=6)
    th2 = np.linspace(3*np.pi/8, 5*np.pi/8, 100)
    disp2_grd, th2_grd = field(dyn.flow, th2)
    disp2_grd_cm = disp2_grd/500
    x2_grd = np.meshgrid(x, th2)[0]
    #cb2 = inset.contourf(th2_grd, x2_grd, disp2_grd_cm, vmin=vmin, vmax=vmax, levels=500,cmap='inferno')
    cb2 = inset.contourf(th2_grd, x2_grd, disp2_grd_cm, levels=500,cmap='inferno')
    inset.set_theta_zero_location('N')
    inset.yaxis.grid(False)
    inset.xaxis.grid(False)
    inset.axis("off")
    inset.set_yticklabels([])
    inset.set_xticks([])
    inset.set_ylim([0,1])
    inset.set_thetamin(80)
    inset.set_thetamax(100)
    #colorb2 = fig.colorbar(cb2)
    #colorb2.set_label('m')
    '''
    if save:
        plt.savefig('/Users/benja/Documents/projects/europa/results/xir_Rc{}p_L{}_N{}{}.png'.format(int(eta*100), np.max(L), N, label), dpi=1200)
    else:

        plt.show()

my_plot1()
my_plot2()



plt.figure(figsize=(4,4))
[ plt.plot(l, np.log10( abs(dyn.k_hydro(l)*dyn.Ulm(l, 2)/dyn.Ulm(2,2)) ), 'ok') for l in dyn.degree]
plt.plot(dyn.degree, np.log10(abs(np.real(dyn.phi))/dyn.Ulm(2,2)), 'ok', mfc='w', alpha=0.5)
plt.xlabel(('degree, $\ell$'))
plt.ylabel((r"Gravity, $|\phi_{\ell m}'|/U_{22}$ "))
plt.xlim((0,100))
plt.ylim((-15,0))
my_plt_opt()
plt.show()





