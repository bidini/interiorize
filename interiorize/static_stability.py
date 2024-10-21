# Integration by fitting chebyshev polynomials.
# An n=1 polytrope with a compositional gradient (static stability) is evaluated. 
#
# Ben Idini, Apr 2020.

import numpy as np
from scipy.special import spherical_jn as jn
from scipy.integrate import cumtrapz
from scipy.special import factorial
from scipy.special import lpmv as Plm
import matplotlib.pyplot as plt
from interiorize.solvers import cheby
import pdb

import subprocess
import pygyre as pg
from os import system,listdir,path

from matplotlib import rcParams
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
rcParams['mathtext.fontset'] = 'cm'

class HydroTides:
    # Solve the problem of hydrostatic tides from a density profile
    # Some issues here. But I am not using this.

    def __init__(self,rho,g,p,r,l):
        self.rho = rho
        self.g = g
        self.drho_dr = np.gradient(rho,r)
        self.dp_dr = np.gradient(p,r)
        self.degree = l

    def f(self,xi):
        return np.zeros(len(xi))

    def p(self,xi):
        return xi**2

    def q(self,xi):
        return 2*xi

    def r(self,xi):
        l = self.degree
        K = 2.1e12
        return -l*(l+1) + xi**2*self.rho*self.drho_dr/self.dp_dr * 2*K

class SlowTides:
    # Solve the low-frequency tide, compared to the dynamical frequency of the planet.
    # In practice, the tidal frequency is assumed to be zero.

    def __init__(self, rho, Z, N, g, degree):
        self.rho = rho
        self.Z = Z
        self.g = g # rads/sec2
        self.N = N
        self.l = degree

    def f(self,xi):
        return np.zeros(len(xi)) 

    def p(self,xi):
        return xi**2

    def q(self,xi):
        return 2*xi 

    def r(self,xi):
        l = self.l
        G = 6.67e-8
        return -l*(l+1) + xi**2 * ( ((1-0.4*self.Z)/(1-self.Z))**2 + 4*np.pi*G*self.rho*self.N**2/self.g**2 ) 

class SlowTides2:
    # Solve the low-frequency tide, compared to the dynamical frequency of the planet.
    # In practice, the tidal frequency is assumed to be zero.

    def __init__(self, rho, Y, N, g, degree):
        self.rho = rho
        self.Y = Y
        self.g = g # rads/sec2
        self.N = N
        self.l = degree

    def f(self,xi):
        return np.zeros(len(xi)) 

    def p(self,xi):
        return xi**2

    def q(self,xi):
        return 2*xi 

    def r(self,xi):
        l = self.l
        G = 6.67e-8
        return -l*(l+1) + xi**2 * ( (.851*(1-0.55*self.Y)/(1-self.Y))**2 + 4*np.pi*G*self.rho*self.N**2/self.g**2 ) 

class SlowTides3:
    # Solve the low-frequency tide, compared to the dynamical frequency of the planet.
    # In practice, the tidal frequency is assumed to be zero.
    # fy comes from a EOS in the form p = K*rho^2*fy^2

    def __init__(self, rho, fy, N, g, degree,Gm_1):
        self.rho = rho
        self.g = g 
        self.N = N
        self.l = degree
        self.Gm_1 = Gm_1
        self.fy = fy

    def f(self,xi):
        return np.zeros(len(xi)) 

    def p(self,xi):
        return xi**2

    def q(self,xi):
        return 2*xi

    def r(self,xi):
        l = self.l
        K = 2.1e12
        return -l*(l+1) + xi**2*( 2/self.Gm_1/self.fy**2 + 2*K*self.rho*(self.N/self.g)**2 ) 

class FullTides:
    # Solve the problem were the tidal frequency is comparable to the dynamical frequency of the planet.
    # Dynamical tides are on. The Coriolis effect is on.
    def __init__(self,cheby,l=[2,4,6],m=2,p=None,rho=None,Z=None,N=None,g=None,om=0,OM=0,eta=0,a=1e8,ms=1e25,R=1e10,K=2.1e12,gamma1=2):
    
        # physical quantities
        self.order = m
        self.degree = l
        self.p = p # pressure profile
        self.rho = rho # density profile
        self.Z = Z # profile of heavy elements
        self.N = N # profile of Brunt-Vaisala frequency
        self.g = g # profile of gravitational acceleration
        self.om = om # tidal frequency
        self.OM = OM # spin rate
        self.eta = eta # Bulk kinematic viscosity
        self.a =  a # satellite semi-major axis
        self.ms = ms # satellite mass
        self.R = R # planetary radius
        self.K = K # EOS constant
        self.gamma1 = gamma1
        self.G = 6.67e-8 # Gravitational constant
        
        # solver parameters
        self.n = np.arange(0,cheby.N)
        self.cheby = cheby
        
    def Ulm(self,l):
        m = self.order
        if l>=2:
            return self.G*self.ms/self.a*(self.R/self.a)**l *np.sqrt(4*np.pi*factorial(l-m)/(2*l+1)/factorial(l+m))*Plm(m,l,0)
        else:
            return 0
    
    # Equation 1: continuity
    def eq1_R(self,l):
        arr = []
        [arr.append( self.cheby.dTn_x(n) + (2/self.cheby.xi-1/self.gamma1/self.p*self.rho*self.g)*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T 

    def eq1_S(self,l):
        arr = []
        [arr.append( -l*(l+1)*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T

    def eq1_pp(self,l):
        arr = []
        [arr.append( 1/self.gamma1/self.p*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T

    def eq1(self,l):
        ZB = np.zeros((len(self.cheby.xi),len(self.n)))
        arr = np.concatenate( (self.eq1_R(l), self.eq1_S(l), ZB, self.eq1_pp(l), ZB), axis=1)
    
        #replace the bc: R = 0 at the origin
        zb = np.zeros(len(self.n))
        bc1 = []
        [bc1.append( self.cheby.Tn_bot(n) ) for n in self.n]
        bc1 = np.concatenate( (bc1,zb,zb,zb,zb ) )
        arr[-1,:] = bc1
        #replace the bc: no bulk stress at the surface:
        bc2 = []
        [bc2.append( self.cheby.dTndx_top(n) + 2/self.R*self.cheby.Tn_top(n) ) for n in self.n]
        bc3 = []
        [bc3.append( -l*(l+1)*self.cheby.Tn_top(n) ) for n in self.n]
        bc = np.concatenate( (bc2,bc3,zb,zb,zb ) )
        arr[-1,:] = bc
        return arr

    # Equation 2: Poisson
    def eq2_R(self,l):
        arr = []
        [arr.append( 4*np.pi*self.G*self.rho*self.N**2/self.g*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T

    def eq2_pp(self,l):
        arr = []
        [arr.append( 4*np.pi*self.G*self.rho/self.gamma1/self.p*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T

    def eq2_phip(self,l):
        arr = []
        [arr.append( self.cheby.d2Tn_x2(n) + 2/self.cheby.xi*self.cheby.dTn_x(n) -l*(l+1)/self.cheby.xi**2*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T
    
    def eq2(self,l):
        ZB = np.zeros((len(self.cheby.xi),len(self.n)))
        arr = np.concatenate( (self.eq2_R(l), ZB, ZB, self.eq2_pp(l), self.eq2_phip(l)), axis=1)
        #replace the bc: gravity potential.
        zb = np.zeros(len(self.n))
        bc1 = []
        [bc1.append( (l+1)*self.cheby.Tn_top(n) + self.R*self.cheby.dTndx_top(n) ) for n in self.n]
        bc1 = np.concatenate( (zb,zb,zb,zb,bc1 ) )
        bc2 = []
        [bc2.append( -l/self.cheby.loend*self.cheby.Tn_bot(n) + self.cheby.dTndx_bot(n) ) for n in self.n]
        bc2 = np.concatenate( (zb,zb,zb,zb,bc2 ) )
        arr[-1,:] = bc1
        arr[0,:] = bc2
        return arr 

    # Equation 3: momentum radial component
    def eq3_R(self,l):
        arr = []
        [arr.append( (-self.om**2 + self.N**2 +1j*self.eta*self.om*(l*(l+1)+2)/self.cheby.xi**2)*self.cheby.Tn(n) - 2j*self.eta*self.om/self.cheby.xi*self.cheby.dTn_x(n) + 1j*self.eta*self.om*self.cheby.d2Tn_x2(n) ) for n in self.n ]
        return np.array(arr).T

    def eq3_S(self,l):
        arr = []
        [arr.append( 2*self.om*self.OM*self.order*self.cheby.xi*self.cheby.Tn(n) + 1j*self.eta*self.om*l*(l+1)*self.cheby.dTn_x(n) ) for n in self.n ]
        return np.array(arr).T

    def eq3_pp(self,l):
        arr = []
        [arr.append( (self.g/self.gamma1/self.p)*self.cheby.Tn(n) + 1/self.rho*self.cheby.dTn_x(n) ) for n in self.n ]
        return np.array(arr).T

    def eq3_phip(self,l):
        arr = []
        [arr.append( self.cheby.dTn_x(n) ) for n in self.n ]
        return np.array(arr).T
    
    def eq3(self,l):
        ZB = np.zeros((len(self.cheby.xi),len(self.n)))
        arr = np.concatenate( (self.eq3_R(l), self.eq3_S(l), ZB, self.eq3_pp(l), self.eq3_phip(l)), axis=1)
        #replace the bc: pp = 0 at surface
        zb = np.zeros(len(self.n))
        bc1 = []
        [bc1.append( self.cheby.Tn_top(n) ) for n in self.n]
        bc1 = np.concatenate( (zb,zb,zb,bc1,zb ) )
        arr[-1,:] = bc1
        return arr 

    # Equation 4: divergence momentum angular component
    def eq4_R(self,l):
        arr = []
        [arr.append( (2j*self.eta*self.om/self.cheby.xi - 2*self.om*self.OM*self.order*self.cheby.xi/l/(l+1))*self.cheby.Tn(n) + 1j*self.eta*self.om*self.cheby.dTn_x(n) ) for n in self.n ]
        return np.array(arr).T

    def eq4_S(self,l):
        arr = []
        [arr.append( (self.om**2*self.cheby.xi**2 - 2*self.om*self.OM*self.order*self.cheby.xi**2/l/(l+1) -1j*self.eta*self.om*l*(l+1) )*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T

    def eq4_pp(self,l):
        arr = []
        [arr.append( -1/self.rho*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T

    def eq4_phip(self,l):
        arr = []
        [arr.append( self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T
    
    def eq4(self,l):
        ZB = np.zeros((len(self.cheby.xi),len(self.n)))
        arr = np.concatenate( (self.eq4_R(l), self.eq4_S(l), ZB, self.eq4_pp(l), self.eq4_phip(l)), axis=1)
        return arr 
    
    # Equation 5: curl  momentum angular component
    def eq5_T(self,l):
        arr = []
        [arr.append( (1j*self.eta*self.om*l**2*(l+1)**2/self.cheby.xi**2 - self.om**2*l*(l+1) + 2*self.om*self.OM*self.order )*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T

    def eq5(self,l):
        ZB = np.zeros((len(self.cheby.xi),len(self.n)))
        arr = np.hstack( (ZB, ZB, self.eq5_T(l), ZB, ZB) )
        #replace the bc: finiteness of Sl at the origin
        zb = np.zeros(len(self.n))
        bc1 = []
        [bc1.append( 1j*self.eta*self.om*l*(l+1)*self.cheby.Tn_bot(n) ) for n in self.n]
        bc2 = []
        [bc2.append( 1/self.rho[0]*self.cheby.Tn_bot(n) ) for n in self.n]
        bc3 = []
        [bc3.append( -self.cheby.Tn_bot(n) ) for n in self.n]
        bc = np.concatenate( (zb,bc1,zb,bc2,bc3) )
        arr[0,:] = bc
        return arr 
    

    # =========
    # Couplings
    def Q(self,l):
        m = self.order
        if l>m:
            return np.sqrt( (l**2 - m**2)/(4*l**2-1) )
        else:
            return 0

    # Equation 3: momentum radial component
    def eq3_Tup(self,l):
        arr = []
        [arr.append( -2j*self.om*self.OM*self.cheby.xi*self.Q(l+1)*(l+2)*self.cheby.Tn(n)  ) for n in self.n ]
        return np.array(arr).T

    def eq3_Tdown(self,l):
        arr = []
        [arr.append( 2j*self.om*self.OM*self.cheby.xi*self.Q(l)*(l-1)*self.cheby.Tn(n)  ) for n in self.n ]
        return np.array(arr).T

    # Equation 4: divergence momentum angular component
    def eq4_Tup(self,l):
        arr = []
        [arr.append( -2j*self.om*self.OM*self.cheby.xi**2*self.Q(l+1)*(l+2)/(l+1)*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T

    def eq4_Tdown(self,l):
        arr = []
        [arr.append( -2j*self.om*self.OM*self.cheby.xi**2*(l+1)*self.Q(l)/l*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T
    
    # Equation 5: curl  momentum angular component
    def eq5_Rup(self,l):
        arr = []
        [arr.append( -2j*self.om*self.OM/self.cheby.xi*self.Q(l+1)/l*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T

    def eq5_Rdown(self,l):
        arr = []
        [arr.append( 2j*self.om*self.OM/self.cheby.xi*self.Q(l)*(l+1)*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T

    def eq5_Sup(self,l):
        arr = []
        [arr.append( -2j*self.om*self.OM*self.Q(l+1)*l*(l+2)*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T

    def eq5_Sdown(self,l):
        arr = []
        [arr.append( -2j*self.om*self.OM*self.Q(l)*(l-1)*(l+1)*self.cheby.Tn(n) ) for n in self.n ]
        return np.array(arr).T
    # ========
    
    # assamble L matrix for given degree
    def MediumL(self,l):
        return np.vstack( (self.eq1(l), self.eq2(l), self.eq3(l), self.eq4(l), self.eq5(l)) )

    def MediumLup(self,l):
        ZB = np.zeros((len(self.cheby.xi),len(self.n)))
        ll1 = np.hstack( (ZB,ZB,ZB,ZB,ZB) )
        ll2 = ll1
        ll3 = np.hstack( (ZB,ZB,self.eq3_Tup(l),ZB,ZB) )
        ll3[-1,:] *= 0
        ll4 = np.hstack( (ZB,ZB,self.eq4_Tup(l),ZB,ZB) )
        ll5 = np.hstack( (self.eq5_Rup(l),self.eq5_Sup(l),ZB,ZB,ZB) )
        ll5[0,:] *= 0
        return np.vstack( (ll1,ll2,ll3,ll4,ll5) )
    
    def MediumLdown(self,l):
        ZB = np.zeros((len(self.cheby.xi),len(self.n)))
        ll1 = np.hstack( (ZB,ZB,ZB,ZB,ZB) )
        ll2 = ll1
        ll3 = np.hstack( (ZB,ZB,self.eq3_Tdown(l),ZB,ZB) )
        ll3[-1,:] *= 0
        ll4 = np.hstack( (ZB,ZB,self.eq4_Tdown(l),ZB,ZB) )
        ll5 = np.hstack( (self.eq5_Rdown(l),self.eq5_Sdown(l),ZB,ZB,ZB) )
        ll5[0,:] *= 0
        return np.vstack( (ll1,ll2,ll3,ll4,ll5) )

    # assemble Big L matrix for degrees up to truncation L
    def BigL(self):
        L = self.degree
        ZB = np.zeros((5*len(self.cheby.xi),5*len(self.n)))
        
        first_row = np.hstack( (self.MediumL(L[0]),self.MediumLup(L[0]),np.hstack([ZB]*(len(L)-2))  ) ) 
        second_row = np.hstack( (self.MediumLdown(L[1]), self.MediumL(L[1]),self.MediumLup(L[1]),np.hstack([ZB]*(len(L)-3))  ) )
        seclast_row = np.hstack( (np.hstack([ZB]*(len(L)-3)), self.MediumLdown(L[-2]),self.MediumLdown(L[-2]),self.MediumLup(L[-2]) ) ) 
        last_row = np.hstack( (np.hstack([ZB]*(len(L)-2)), self.MediumLdown(L[-1]),self.MediumL(L[-1]) ) ) 
        middle_rows = []
        [middle_rows.append( np.hstack( (np.hstack([ZB]*(row-1)),self.MediumLdown(L[row]), self.MediumL(L[row]),self.MediumLup(L[row]),np.hstack([ZB]*(len(L)-2-row))) )  ) for row in range(2,len(L)-2)]

        self.BigL = np.vstack( (first_row,second_row,np.vstack((middle_rows)),seclast_row,last_row) )
        return

    # assamble Big f vector for degrees up to truncation L
    def Bigf(self):
        L = self.degree
        self.Bigf = np.zeros(len(self.cheby.xi)*5*len(self.degree))
        for i in range(0,len(self.degree)):
            self.Bigf[len(self.cheby.xi)*5*i+len(self.cheby.xi)+1] = (2*L[i]+1)*self.Ulm(L[i]) # check the indexing.
        return

    def solve(self):
        self.BigL()
        self.Bigf()
        a = self.cheby.lin_solve(self.BigL, self.Bigf)
        spa = np.split(a, len(self.n))
        
        self.R = []
        self.S = []
        self.T = []
        self.pp = []
        self.phip = []
        [self.R.append(np.split(arr,5)[0]) for arr in spa]
        [self.S.append(np.split(arr,5)[1]) for arr in spa]
        [self.T.append(np.split(arr,5)[2]) for arr in spa]
        [self.pp.append(np.split(arr,5)[3]) for arr in spa]
        [self.phip.append(np.split(arr,5)[4]) for arr in spa]

        return

#################### 
class HeavyElements:
    # Get profiles of important quantities assuming that the abundace of heavy elements is a perturbation over the EOS of an index-one polytrope. 
    # Zc:    abundance of heavy elements in the inner core.
    # Ze:    abundance of heavy elements in the envelope.
    # Lc:    size in radius of the dilute core.
    # xi:    collocation points along the radius
    # xic:   dilute core inner boundary
    # xr:   contraint to the planetary radius (rad). Controls the total mass of heavy elements.
    # Me:   Earth's mass. Only used to report results
    # Success is False if the maximum number of iterations is reached.
    
    def __init__(self,rhom=1,Zc=0,Ze=0,xr=3,Lc=0,K=2.1e12,G=6.67e-8,Me=5.97e27):
        self.init_param = {'rhom': rhom,
                'Zc': Zc,
                'Ze': Ze,
                'xr': xr,
                'Lc': Lc}

        self.constants = {'K': K,
                'k': np.sqrt(2*np.pi*G/K),
                'G': G,
                'Me': Me}

        # fields to calculate
        self.init_cheb = []
        self.update_cheb = []
        self.xi = []
        self.rho = []
        self.Z = []
        self.xic = [] # inner core radius
        self.xoc = [] # outer core radius
        self.dir_gyre_model = []
        self.error = ''

    def init_elements(self, N=100, a=1e-6, b=np.pi):
        #iniate spectral elements
        self.init_cheb = cheby(npoints=N, loend=a, upend=b, direction=-1)
        return

    def get_interior(self, poly=False, kind='sin', kind2='Z'):
        # Calculate density, pressure, etc from hydrostatic equillibrium.
        
        # load model parameters
        keys = ['rhom','Zc','Ze','Lc','xr']
        rhom, Zc, Ze, Lc, xr = [self.init_param.get(key) for key in keys] 
        
        k = self.constants['k']
        
        # Get density, heavy elements. 
        self.xi, self.rho, self.Z, self.xic, self.success = self.get_radius_rho_Z(self.init_cheb.xi,rhom,Zc,Ze,Lc,xr,kind,poly,kind2=kind2)

        # dilute core outer radius
        self.xoc = self.xic + Lc
        
        # update solver to xi
        self.update_cheb = cheby(npoints=self.init_cheb.N, loend=min(self.xi), upend=max(self.xi), direction=-1)

        # Radial coordinate
        self.r = self.xi/k

        # gravity acceleration
        self.g = self.get_g(self.r, self.rho, self.constants['G'])

        # Pressure
        self.p = self.get_p(self.r,self.rho*self.g)

        # Brunt-Vaisala frequency
        self.N2 = self.get_brunt_vaisala(self.g, self.rho, self.p, self.r)

        # total abundance of heavy elements in earth masses
        self.Mz =  4*np.pi*np.trapz(self.Z*self.xi**2*self.rho,x=self.xi)/k**3/self.constants['Me'] # 

        # planetary radius
        self.R = self.r[-1]

        # accumulated planetary mass
        self.M_cum = 4*np.pi*cumtrapz(self.r**2*self.rho,x=self.r,initial=4/3*np.pi*self.r[0]**3*self.rho[0])

        # planetary mass
        self.Mt = self.M_cum[-1]

        # dynamical frequency
        self.omd = np.sqrt(self.constants['G']*self.Mt/self.R**3)

        return
    
    def get_hydroLove(self,l=2):
        # Calculate hydrostatic tidal response.
        # It requires a defined interior model
        cheb = self.update_cheb 
        k = self.constants['k']

        tide = HydroTides(self.rho, self.g, self.p, self.r,l)
        cheb.solve(tide.p, tide.q, tide.r, tide.f, [1,-l/min(self.xi),0], [max(self.xi),l+1,(2*l+1) ])
        phi_hydro = cheb.u(cheb.T)[0] - (self.r/self.R)**l
        love = phi_hydro[-1]

        return love

    def plot_interior(self):
        fig=plt.figure(figsize=(4,4))
        plt.plot(self.r/self.r[-1], self.rho/10, label=r'$\rho\cdot10^{-1} $')
        plt.plot(self.r/self.r[-1], self.Z, label=r'$Z$')
        plt.plot(self.r/self.r[-1], self.g/10000, label=r'$g\cdot10^{-4}$')
        plt.plot(self.r/self.r[-1], self.N2/self.omd**2, label=r'$N^2/\Omega_{dyn}^2$')
        plt.minorticks_on()
        plt.tick_params(which='both',direction='in',top=True,right=True)
        for tick in plt.xticks()[1]+plt.yticks()[1]:
            tick.set_fontname("DejaVu Serif")
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.xlabel(('Normalized radius, $r/R$'))
        plt.ylabel((r'cgs'))
        plt.legend()
        plt.tight_layout()
        plt.show()
        return 
    
    def get_radius_rho_Z(self,xi,rhom,Zc,Ze,Lc,xr,kind,poly,kind2):
        if poly:
            xi, rhon, Z, xic, success = xi, jn(0,xi), np.zeros(len(xi)), 0,True
        else:
            xi, rhon, Z, xic, success = self.iter_rho(xi,Zc,Ze,Lc,xr,kind,kind2)
        rhoc = (rhom*xi[-1]**3)/(3*np.trapz(xi**2*rhon, x=xi))
        return xi, rhon*rhoc, Z, xic, success

    def iter_rho(self,xi,Zc,Ze,Lc,xr,kind,kind2):
        # xr: target planetary radius
        # Lc: width of grad.Z
        # Zc, Ze: Z in core and envelope

        # implement the radius constraint
        a = xi[0] 
        b = xi[-1] # iteration planetary radius
        xin = xi/b
        if kind2 is 'Z':
            new_xic = 0.01*b # iteration inner boundary grad.Z
        #new_xic =0 
            db = 2 # controls how the radius changes each iteration
        else:
            new_xic = 0.78*b
            db = 1
        ite = 0
        ite_max = 400
        success = True
        # Iterate inner core boundary until planetary radius b matches the target xr
        while abs(b-xr)>0.0001*xr and ite < ite_max:
            xic = new_xic
            Z = self.get_Z(xi,Zc,Ze,xic,Lc,kind=kind) # initial grid of Z 
            
            rhon = self.edo_rho(Z, xi,kind2) # initial guess normalized density
            # Iterate planetary radius until density becomes close to zero at planet radius 
            while abs(rhon[np.argmax(xi)]) > 0.0001 and ite < ite_max:
                new_b = b + rhon[np.argmax(xi)]*db
                xi = xin*new_b
                Z = self.get_Z(xi, Zc, Ze, xic, Lc, kind=kind)
                rhon = self.edo_rho(Z, xi,kind2)
                b = new_b # update radius
                ite += 1

            rhon -= rhon[-1]
            rhon += 1e-4 # approximately 1 mg/cc atmospheric density
            new_xic = xic *(1+ 10*(b-xr)/xr) 
            ite += 1

        if ite >= ite_max:
            success = False

        Z = self.get_Z(xi, Zc, Ze, new_xic, Lc, kind=kind)

        return xi, rhon, Z, new_xic, success

    # rho_c:    central density
    def edo_rho(self, Z, xi, kind='Z'):
        # Constant ratio of H-He and Z density model by me.
        # return: non dimensional density model scaled to the central density
        if kind is 'Z': 
            rho_rhoz = 0.42*jn(0,xi)**0.55
            f_z = (1-Z)/(1-rho_rhoz*Z)
            Dlnf_z = np.gradient(np.log(f_z), xi) 
        elif kind is 'Y':
            f_z = (1-Z)/(1-0.53*Z)/(1-.28)
            f_z = (1-Z)/(1-0.53*Z)/.851
            Dlnf_z = np.gradient(np.log(f_z), xi) 
        elif kind is 'Z_v2':
            rho_rhoz = 0.4
            f_z = (1-Z)/(1-0.4*Z)
            Dlnf_z = np.gradient(np.log(f_z), xi) 

        # define the ODE:
        def p(xi):
            return np.ones(len(xi))

        def q(xi):
            return  Dlnf_z + 2/xi

        def r(xi):
            return 1/f_z**2

        def f(xi):
            return np.zeros(len(xi))
        
        def L(p, q, r, clo, cup):
            L = []
            [L.append( p*sol.d2Tn_x2(n) + q*sol.dTn_x(n) + r*sol.Tn(n, sol.T) ) for n in np.arange(0, sol.N)]
            return np.vstack( (np.array(L).T, bc(clo, kind='lower'), bc(cup, kind='lower')) ) # add the boundary conditions ('ivp')

        def bc(c, kind='lower'):
            bc = []
            if kind is 'lower':
                [bc.append( c[0]/sol.dxi_dx*(-1)**(n+1)*n**2 + c[1]*(-1)**n ) for n in np.arange(0, sol.N)]
            elif kind is 'upper':
                [bc.append( c[0]/sol.dxi_dx*n**2 + c[1] ) for n in np.arange(0, sol.N)]
            return bc 

        def u(t,an):
            u = []
            [u.append(sol.Tn(n,t)) for n in np.arange(0, sol.N)]
            return np.dot(np.array(u).T, an)

        # Solve the ODE
        # We define a new solver to get the appropriate xi and chain rule
        sol = cheby(npoints=len(xi)+2, loend=min(xi), upend=max(xi), direction=-1)
        # Reworking this
        #sol.solve(p, q, r, f, [0,1,min(f_z)], [1,0,0 ], kind='ivp') 
        p = p(xi)
        q = q(xi)
        r = r(xi)
        cup = [1, 0, 0 ]
        clo = [0, 1, min(f_z)]
        blockL = L(p, q, r, clo, cup)
        f_bc = np.concatenate( (f(xi), [clo[2], cup[2]]) ) 
        an = sol.lin_solve( blockL, f_bc) 
        
        return u(sol.T,an)/f_z

    # Get the normalized g (see the normalization factor in my notes)
    def get_g(self,x,rho_x,G):
        g = 4*np.pi*G*cumtrapz(x**2*rho_x, x=x,initial=G*4/3*np.pi*x[0]**3*rho_x[0])/x**2 
        return g 
    
    def get_p(self,x,rhog):
        p = -cumtrapz(rhog[::-1], x=x[::-1],initial=-1e6)[::-1]
        return p

    # dimensional BV frequenct
    def get_brunt_vaisala(self, g, rho, p, r):
        dp = np.gradient(p,r)
        drho = np.gradient(rho,r)
        N2 = g*(0.5*dp/p-drho/rho)
        N2[np.isnan(N2)]=0
        return N2 
    
    def get_Z(self,xi,Zc, Ze, xic, Lc,kind='tanh'): 
        if kind is 'tanh':
            return Ze + (Zc-Ze)/2*( 1-np.tanh(2*np.pi/Lc*(xi-xic) - np.pi ) )

        elif kind is 'sin':
            return Ze + (Zc-Ze)*np.sin( np.pi/2*(xic+Lc-xi)/Lc)**2*(xi>xic)*(xi< xic+Lc) + (Zc-Ze)*(xi<xic)


    def run_gyre(self, dir_model, in_gyre, bin_gyre):
        self.dir_gyre_model = dir_model 
        self.write_gyre_model(outdir=dir_model)

        p = subprocess.Popen([bin_gyre, in_gyre],cwd=dir_model)

        p.wait()
        return
    
    def process_gyre_mode(self, mode):

        file_mode = f'{self.dir_gyre_model}/{mode.get("fname")}' 
        if (not path.exists(file_mode)):
            print('Error: Gyre mode {} not found.'.format(mode.get("fname")))
            return 'Error: Gyre mode {} not found.'.format(mode.get("fname"))

        dt = pg.read_output(file_mode)
        mode['om0'] = np.real(dt.meta['omega'])
        keys = ['xi_r','xi_h','eul_rho','rho','x','eul_phi']
        [mode.setdefault(key, dt[key]) for key in keys]        
        mode['l'] = dt.meta['l']
        mode['m'] = dt.meta['m']
        
        rho_norm_ct = np.trapz(mode['x']**2*mode['rho'],x=mode['x'])*4*np.pi
        rho_hat = mode['rho']/rho_norm_ct

        mode['C'] = 4*np.pi*np.trapz( rho_hat*mode['x']**2*(2*mode['xi_r']*mode['xi_h']+mode['xi_h']**2) ,x=mode['x'] )
        mode['Q'] = 4*np.pi*np.trapz( mode['x']**(mode['l']+2)*np.conj(mode['eul_rho']*rho_hat) , x=mode['x'] )
        # NOTE: Q should be multipled by OMEGA/omd to get the same numbers in Lai (2021).
        # Detail on data and units: https://gyre.readthedocs.io/en/stable/ref-guide/output-files/detail-files.html#stellar-structure

        return mode

    def get_dynLove(self,mode,omega, OMEGA=2*np.pi/(9.9*3600)):
        # Attention that the normalization of 'omega' should agree with the normalization of Q and the other frequencies.
        # In the current form, 'omega' goes from 0 to 2.
        k = 4*np.pi/(2*mode['l']+1)*mode['Q']**2/( (mode['om0']*self.omd/OMEGA)**2-(mode['m']*mode['C']+omega)**2)
        return k

    def write_gyre_model(self, downsample=False, outdir=None):
        " Based on Chris Mankovich 'kronos'"
        brunt_key = 'N'

        vec = {'l': self.r}

        n = downsample if downsample else len(vec['l'])
        mtot = self.Mt
        rtot = self.R
        ltot = 1.
        version = 101 # see format spec in gyre-5.2/doc/mesa-format.pdf

        # ['l', 'req', 'rpol', 'rho', 'p', 'm_calc', 'y', 'z', 't', 'gamma1', 'grada', 'g', 'dlnp_dr', 'dlnrho_dr', 'gradt', 'n2']
        vec['k'] = np.arange(n)
        vec['gamma1'] = 2.0*np.ones(n)
        #dpdr = np.diff(self.p,prepend=0)
        #drhodr = np.diff(self.rho,prepend=0)
        #vec['gamma1'] = self.rho/self.p*dpdr/drhodr
        vec['rho'] = self.rho
        vec[brunt_key] = self.N2
        vec['m_calc'] = self.M_cum
        vec['p'] = self.p

        # irrelevant in adiabatic calculations
        vec['delta'] = np.zeros(n) # polytropic model
        vec['gradt'] = (vec['gamma1']-1)/vec['gamma1']
        vec['grada'] = vec['gradt']
        vec['t'] = 10**(1/2*1e-6*128**2 + 1/2*np.log10(self.p))
        vec['lum'] = np.ones(n) # adiabatic mode calculation doesn't need luminosity
        vec['omega'] = np.zeros(n)
        if outdir is None:
            outfile = f'model.gyre'
        else:
            outfile = f'{outdir}/model.gyre'

        ki = []
        li = []
        jumps = []

        '''
        # if downsampling, must resample all quantities other than k, lum, omega
        new_l = np.linspace(vec['l'][0], vec['l'][-1], n)
        brunt_key = {
            'default':'n2',
            'equatorial':'n2eq',
            'direct':'n2_direct'
        }[brunt_option]
        for qty in 'm_calc', 'p', 't', 'rho', 'gradt', brunt_key, 'gamma1', 'grada', 'delta':
            vec[qty] = splev(new_l, splrep(vec['l'], vec[qty], k=1)) # was k=3 previously, changed 02222021
        vec['l'] = new_l
        '''

        n = len(vec['k'])
        # now set header. n may have been updated if density discontinuities are present.
        header = '{:>5n} {:>16.8e} {:>16.8e} {:>16.8e} {:>5n}\n'.format(n, mtot, rtot, ltot, version)

        with open(outfile, 'w') as fw:
            fw.write(header)

            ncols = 19
            for k in vec['k']:
                data_fmt = '{:>5n} ' + '{:>16.8e} ' * (ncols - 1) + '\n'

                if vec[brunt_key][k] < 1e-12: vec[brunt_key][k] = 1e-12 # 07202020: previously had 0, which can give gyre a hard time # reduce to 1e-20 02222021
                if vec[brunt_key][k] < 0.: vec[brunt_key][k] = 1e-12 # 08252020 # 02222021

                data = k+1, vec['l'][k], vec['m_calc'][k], vec['lum'][k], vec['p'][k], vec['t'][k], vec['rho'][k], \
                    vec['gradt'][k], vec[brunt_key][k], vec['gamma1'][k], vec['grada'][k], vec['delta'][k], \
                    1, 0, 0, 1, 0, 0, \
                    vec['omega'][k]
                fw.write(data_fmt.format(*data))
