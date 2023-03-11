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
N2      = 1e-7         # Ocean stratification

# Jupiter
Mj = 1.898e30
Rj = 6.99e9
tau = 1e6

# Chebyshev solver
N = 150         # number of Chebyshev polynomialsi
Lmax = 100
M = 2
save = False
label = 'tau6'

L = np.arange(abs(M), Lmax+1)

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

#eta     = np.linspace(0.8,0.81,100)
#eta     = np.linspace(0.87,0.88,100)
#eta     = np.linspace(0.88,0.89,100)
#eta     = np.linspace(0.9,0.91,100)
#eta     = np.linspace(0.91,0.95,100)
#eta     = np.linspace(0.8705,0.871,100)
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
#ome = 1.2e-5 # NOTE: an inertial mode resonance

print('conventional tidal frequency {} mHz'.format(om*1e3))
print('eccentricity tidal frequency {} mHz'.format(ome*1e3))

'''
## ITERATE OVER OCEAN THICKNESS
for i in range(0, len(eta)):
    cheb = cheby(npoints=N, loend=a[i], upend=b)

    dyn = dynamical(cheb, ome, Om, ms, Mj, sma, Rp, 
                rho=rhow, rhoc=rhoc[i], Rc=Rc[i],
                l=L, m=M, tau=tau, x1=a[i], x2=b,
                tides='ee', e=e)

    dyn.solve(kind='bvp-core') # flow with ocean bottom
    print('eta = {:.5f}R     l = 2: {:.4f}%    l = 4: {:.4f}%'.format(eta[i], dyn.dk[0],  dyn.dk[1])) 

exit()
'''

## SOLVE THE PROBLEM AT A SINGLE FREQUENCY

# Initialize the solver tool
cheb = cheby(npoints=N, loend=a, upend=b)

# Set the problem matrix
dyn = dynamical(cheb, ome, Om, ms, Mj, sma, Rp, 
                rho=rhow, rhoc=rhoc, Rc=Rc,
                l=L, m=M, tau=tau, x1=a, x2=b,
                tides='ee', e=e, N2=N2*0)


# Solve the linear problem
dyn.solve(kind='bvp-core') # flow with ocean bottom

# Save a copy of results
if save:
    dill.dump(dyn.an, file=open('/Users/benja/Documents/projects/europa/results/an_Rc{}p_L{}_N{}{}.pickle'.format(int(eta*100), np.max(L), N, label), 'wb') )

##################################################

# Print the percentile dynamic correction to the love number
[print('l = {}: {:.20f}%'.format(L[i], dyn.dk[i]*100 )) for i in range(0,6,2)]

print('log(Q_2): {}'.format(np.log10(abs( dyn.Q[0] ))))

print(dyn.k_hydro(2))

###################################################
# Sanity checks on displacement

# degree 2 displacement at the surface obtained from solving the problem.
xi2 = dyn.y1[0][0]

# Hydrostatic degree-2 displacement at the surface obtained from the boundary condition.
xih2 = (dyn.k_hydro(2)+1)*dyn.Ulm(2,2)/dyn.g

###################################################
# Solve the problem including the forcing
#dyn.forcing = 'forced'
#dyn.solve(kind='bvp') # flow with ocean bottom

#[print('l = {}: {:.20f}%'.format(L[i], dyn.dk[i] )) for i in range(0,3)]

###################################################

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

exit()

plt.figure(figsize=(4,4))
[ plt.plot(l, np.log10( abs(dyn.k_hydro(l)*dyn.Ulm(l, 2)/dyn.Ulm(2,2)) ), 'ok') for l in dyn.degree]
plt.plot(dyn.degree, np.log10(abs(np.real(dyn.phi))/dyn.Ulm(2,2)), 'ok', mfc='w', alpha=0.5)
plt.xlabel(('degree, $\ell$'))
plt.ylabel((r"Gravity, $|\phi_{\ell m}'|/U_{22}$ "))
plt.xlim((0,100))
plt.ylim((-15,0))
my_plt_opt()
plt.show()



exit()

# I need to remove the hydrostatic response to see the effects of dynamical tides
# Write the linear relationship between ocean density and k2 using perturbation theory on the denominator.

##################################################
##################################################
# CALCULATE K2 AS A FUNCTION OF FREQUENCY

def get_om_vs_dk2():
    ome_vec = np.linspace(0.5, 1.5, 100)*ome
    dk2_vec = np.zeros(len(ome_vec))
    dk2_vec2 = np.zeros(len(ome_vec))
    Q_vec = np.zeros(len(ome_vec))
    for i in np.arange(len(ome_vec)):

        print(i)
        sma = (G*(Mj+ms)*(2*np.pi/ome_vec[i])**2/4/np.pi**2)**(1/3)     # satisfy Kepler's third law                   
        Om = ome_vec[i]
        dyn = dynamical(cheb, ome_vec[i], Om, ms, Mj, sma, Rp, 
                    rho=rhow, rhoc=rhoc, Rc=Rc, N2=N2,
                    l=L, m=M, tau=tau, x1=a, x2=b,
                        tides='ee', e=e)

        dyn.solve(kind='bvp') # flow with ocean bottom
        
        dk2_vec[i] = dyn.dk[0] # Love number without salinity

        dyn.forcing = 'forced'
        dyn.solve(kind='bvp') 
        dk2_vec2[i] = dyn.dk[0] # Love number with salinity
        Q_vec[i] = np.log10(abs(dyn.Q[0]))
   
    plt.figure(figsize=(4,4))
    plt.plot(ome_vec/ome, dk2_vec, '-k', label='homogeneous')
    plt.plot(ome_vec/ome, dk2_vec2, '-r', label='salinity gradient')
    plt.xlabel((r'Tidal frequency, $\omega$'))
    plt.ylabel((r'$\Delta k_2$'))
    plt.legend()
#    plt.xlim((0, 2*ome))
#    plt.ylim((-50,50))
    my_plt_opt()

    plt.show()

    return ome_vec, dk2_vec, dk2_vec2, Q_vec

# PLOT RESULTS
def my_plot3(ome_vec, dk2_vec, dk2_vec2):
    plt.figure(figsize=(4,4))
    plt.plot(ome_vec/ome, dk2_vec, '-k', label='homogeneous')
    plt.plot(ome_vec/ome, dk2_vec2, '-r', label='salinity gradient')
    plt.xlabel((r'Tidal frequency, $\omega$'))
    plt.ylabel((r'$\Delta k_2$'))
    plt.legend()
#    plt.xlim((0, 2*ome))
#    plt.ylim((-50,50))
    my_plt_opt()

    plt.show()

a1, a2, a3, a4 = get_om_vs_dk2()
my_plot3(a1, a2, a3)
pdb.set_trace()

################################################################
# PLOT THE DISPLACEMENT FIELD FOR A FEW RESONANCES

# NOTE: a g-mode resonance
ome = 0.9e-5 
N = 100                                # number of Chebyshev polynomialsi
L = np.arange(M if M>1 else 3, M+100, 2)
cheb = cheby(npoints=N, loend=Rc/Rp*np.pi, upend=b)
dyn = dynamical(cheb, ome, Om, ms, Mj, sma, Rp, 
                rho=rhow, rhoc=rhoc, Rc=Rc, N2=N2,
                l=L, m=M, tau=tau, x1=a, x2=b,
                tides='ee', e=e)

dyn.solve(kind='bvp-core') # flow with ocean bottom

[print('l = {}: {:.20f}%'.format(L[i], dyn.dk[i] )) for i in range(0,3)]
#my_plot1()
#my_plot2()

# NOTE: an inertial mode resonance
ome = 2.36e-5 
N = 100                                # number of Chebyshev polynomialsi
L = np.arange(M if M>1 else 3, M+100, 2)
cheb = cheby(npoints=N, loend=Rc/Rp*np.pi, upend=b)
dyn = dynamical(cheb, ome, Om, ms, Mj, sma, Rp, 
                rho=rhow, rhoc=rhoc, Rc=Rc, N2=N2,
                l=L, m=M, tau=tau, x1=a, x2=b,
                tides='ee', e=e)

dyn.solve(kind='bvp-core') # flow with ocean bottom

[print('l = {}: {:.20f}%'.format(L[i], dyn.dk[i] )) for i in range(0,3)]

my_plot1()
my_plot2()

pdb.set_trace()








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

