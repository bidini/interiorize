# Solve the dynamical tides in a Poincare problem (uniform density, full Coriolis). 
#
# Ben Idini, Nov 2022.

import numpy as np
from scipy.special import spherical_jn as jn
from scipy.special import sph_harm as SH

from interiorize.solvers import cheby
from interiorize.poincare import dynamical as inertial
from interiorize.hough import dynamical
import interiorize.plottools as my_plots

from matplotlib import rcParams
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class
from matplotlib.ticker import MaxNLocator

import pdb
import dill

## INPUT PARAMETERS
G   = 6.67e-8                           # gravitational universal constant
# Titan
R       = 2575e5    # Mean radius (cm)
rhom    = 1.8798     # Mean density (g/cc)
MOI     = 0.3414     # Moment of inertia
e       = 0.0288     # Eccentricity
ms      = 1.3452e26    # mass (g)
Ts      = 15.945*24 # orbital period (hours)
# Saturn
Ms = 5.683e29
Rs = 5.8232e9

# MODEL PARAMETERS
rhow    = 1.     
N2      = 1e-8         # Ocean stratification
H       = 350e5     # Ocean bottom depth  (cm)
tau     = 1e9       # frictional dissipation
# Chebyshev solver
N = 80         # number of Chebyshev polynomialsi
Lmax = 80
M = 2
save = True
label = 'tau9_N2_H240-246'

# Initial calculations
L       = np.arange(abs(M), Lmax+1)
eta     = (R-H)/R
Rc      = R*eta        # core radius
Rp      = R             # body radius
a       = np.pi*eta
b       = np.pi                              # planet's surface   
rhoc    = 3*ms/(4*np.pi*Rc**3) - rhow*((Rp/Rc)**3 -1)  # fit satellite's mass
sma     = (G*(Ms+ms)*(Ts*3600)**2/4/np.pi**2)**(1/3)     # satisfy Kepler's third law                   
oms     = 2*np.pi/Ts/3600           # orbital frequency
Om      = oms                # rotational frequency in synchronous rotation
om      = M*(oms - Om)     # conventional tidal frequency
ome     = oms     # eccentricity tidal frequency

print('spin rate: {:.3E}'.format(Om))

# Varying depth
H_vec       = 1e5*np.linspace(240,246, 190) # spacing 0.05e5
eta_vec     = (R-H_vec)/R
Rc_vec      = R*eta_vec
a_vec       = np.pi*eta_vec
rhoc_vec    = 3*ms/(4*np.pi*Rc_vec**3) - rhow*((R/Rc_vec)**3 - 1)

###################################################

# PLOT DEFINITIONS
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
rcParams['mathtext.fontset'] = 'cm'

plot_dir = '/Users/benja/Documents/projects/titan_tides/models_depth/'

def my_plt_opt():

    plt.minorticks_on()
    plt.tick_params(which='both', direction='in', top=True, right=True)
    for tick in plt.xticks()[1] + plt.yticks()[1]:
        tick.set_fontname("DejaVu Serif")

    plt.tight_layout()

    return

## PLOT RESULTS

# Chebyshev spectral coefficients.
# NOTE: gravity at the surface is the sum of the spectral coefficients. If they reach a round-off plateau, we are adding crap to the solution.

def my_plot1():
    plt.figure(figsize=(4,4))
    ans = np.split(dyn.an, 3)
    [ plt.semilogy( np.arange(1,N+1), abs(np.split(ans[2], len(L))[i]), label=r'$\ell = {}$'.format(L[i]), linewidth=1, alpha=0.6) for i in range(0, len(L))] 
    plt.xlim((0,N))
    plt.xlabel(('$n$'))
    plt.ylabel(('$|a_n|$'))
    my_plt_opt()
    if save:
        plt.savefig(plot_dir+'an_Rc{}p_L{}_N{}{}_model{}.png'.format(int(eta*100), np.max(L), N, label, i), dpi=1200)
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
    disp_grd, th_grd = field(dyn.y1, th)
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
    colorb.set_label(r'$\log |\xi_r|$')
    #colorb.set_label(r'm')
    colorb.ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='upper'))
    my_plt_opt()
    if save:
        plt.savefig(plot_dir+'xir_Rc{}p_L{}_N{}{}_model{}.png'.format(int(eta*100), np.max(L), N, label, i), dpi=1200)
    else:

        plt.show()

###################################################

k2_vec = np.zeros(len(H_vec))
E_vec = np.zeros(len(H_vec))

## ITERATE OVER OCEAN STRATIFICATION
for i in range(0, len(H_vec)):
    
    cheb = cheby(npoints=N, loend=a_vec[i], upend=b)

    dyn = dynamical(cheb, ome, Om, ms, Ms, sma, Rp, 
                rho=rhow, rhoc=rhoc_vec[i], Rc=Rc_vec[i],
                l=L, m=M, tau=tau, x1=a_vec[i], x2=b,
                tides='ee', e=e, N2=N2)

    dyn.solve(kind='bvp-core') # flow with ocean bottom

    #my_plot1()
    #my_plot2()
    #plt.close('all')

    k2_vec[i] = dyn.k[0]
    E_vec[i] = sum(dyn.E)
    print('model ', i, ' has a k2 = ',k2_vec[i])
    
dill.dump([H_vec, k2_vec, E_vec], file=open(plot_dir+'k2Q_Rcvec_L{}_N{}{}.pickle'.format(np.max(L), N, label), 'wb') )

pdb.set_trace()
