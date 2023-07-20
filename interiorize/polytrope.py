# Set of ODES defining the tidal response of an index-2 polytrope.
# Solve for the full tide (dynamic+hydrostatic) with the Cowlng approximation

import numpy as np
from scipy.special import spherical_jn as jn
from scipy.special import lpmv as Plm
from scipy.special import factorial
from sympy.physics.quantum.cg import CG
import pdb

class static:
    # Solve the static gravity (benchmark)
    def f(self):
        l = self.degree
        return -(self.cheby.xi/np.pi)**l 

    def p(self):
        return np.ones(len(self.cheby.xi))

    def q(self):
        return 2/self.cheby.xi 

    def r(self):
        l = self.degree
        return -l*(l+1)/self.cheby.xi**2 + 1

class norotation:
    # This is the flow potential without Coriolis. Vorontsov's dynamical tide.
    def __init__(self, cheby,om, Mp, ms,a, Rp, l=2, m=2,b=np.pi,A=4.38,B=0,x0=1e-3,xr=3.14,G=6.67e-8,tides='c'):
        self.degree = l
        self.order = m
        self.om = om
        self.a = a
        self.gravity_factor = G*ms/a
        self.Rp = Rp
        self.rdip = 1
        self.ldamp = 2
        self.G = G
        self.A = 1
        self.B = 0
        self.rhoc = A
        self.g = G*Mp/Rp**2
        self.tides = tides

        self.cheby = cheby
        self.n = np.arange(0,cheby.N)
        self.N = cheby.N
        self.x0 = x0
        self.xr = xr

        self.psi = []
        self.dpsi = []
        self.d2psi = []
        self.an = []
        self.phi = []
        self.phi_dyn = []
        self.dk = []
        self.k = []
        self.Q = []
    
    def rho(self):
        xi = self.cheby.xi
        return (self.A*np.sin(xi) + self.B*np.cos(xi))/xi
        #return (self.A-self.OM/2/np.pi/self.G)*jn(0,xi) + self.OM**2/2/np.pi/self.G*(1-jn(0,xi))

    def drho(self):
        xi = self.cheby.xi
        return ( np.cos(xi)*(self.A*xi-self.B) - np.sin(xi)*(self.A+self.B*xi) )/xi**2
        #return -(self.A-self.OM/2/np.pi/self.G)*jn(1,xi) + self.OM**2/2/np.pi/self.G*jn(1,xi) 

    def Ulm(self, l,m):
        """
        Numerical factor in the tidal forcing.

        kind:
            'c': conventional tides.
            'e': eccentricity tides.
            'o': obliquity tides.
        """
        
        if self.tides is 'c':
            if l >= 2:
                return self.gravity_factor*(self.Rp/self.a)**l *np.sqrt(4*np.pi*factorial(l-m)/(2*l+1)/factorial(l+m))*Plm(m,l,0)
            else:
                return 0

        elif self.tides is 'ee':
            if l >= 2:
                #return self.gravity_factor*(self.Rp/self.a)**l * 42/4*np.sqrt(2*np.pi/15)*self.e
                return self.gravity_factor*(self.Rp/self.a)**l * self.e *(l + 2*m + 1)*np.sqrt(4*np.pi*factorial(l-m)/(2*l+1)/factorial(l+m))*Plm(m,l,0)
            else: 
                return 0
        
    def f(self):
        l = self.degree
        return 5*self.om**2/(4*np.pi*self.G*self.rhoc) * jn(l,self.cheby.xi)

    def p(self):
        return jn(0,self.cheby.xi)

    def q(self):
        return ( 2*jn(0,self.cheby.xi)/self.cheby.xi - jn(1,self.cheby.xi) )

    def r(self):
        l = self.degree
        return - jn(0,self.cheby.xi) * l*(l+1) / self.cheby.xi**2
    
    def u(self):
        u = []
        [u.append(self.cheby.Tn(n)) for n in self.n]
        du = []
        [du.append(self.cheby.dTn_x(n)) for n in self.n]
        d2u = []
        [d2u.append(self.cheby.d2Tn_x2(n)) for n in self.n]
        return (np.dot(np.array(u).T, self.an), np.dot(np.array(du).T, self.an),np.dot(np.array(d2u).T, self.an))

    def solve(self):
        l = self.degree
        m = self.order
        #Bigf = np.concatenate( (self.f, [0, (2*l+1)/self.xr*self.Ulm]) ) # Add the third coefficient in the bc eqn. For the overall potential
        Bigf = np.concatenate( (self.f, [0, 0]) ) # Add the third coefficient in the bc eqn. For the dynamical potential
        #Bigf = np.concatenate( (self.f, [self.Ulm*(2*l+1)/np.pi*jn(l,self.x0)/jn(l-1,np.pi), (2*l+1)/self.xr*self.Ulm]) ) # Add the third coefficient in the bc eqn.
        self.L()
        self.an = self.cheby.lin_solve(self.L, Bigf)
        self.psi, self.dpsi, self.d2psi = self.u()
        
        # Solve the gravity 
        grav = gravity(self.cheby, l=l, m=m,f=self.psi,x0=self.x0,xr=self.xr,Ulm=self.Ulm(l,m) )
        grav.solve()
        self.phi=grav.phi-self.Ulm(l,m)*(self.cheby.xi/self.xr)**l
        #self.phi_dyn.append(grav.phi-self.Ulm(l,m)*(2*l+1)/self.xr*jn(l,self.cheby.xi)/jn(l-1,self.xr) )
        self.grav.phi 
        
        # Love number calculations
        #k = (sum(grav.an)-self.Ulm(l,m))/self.Ulm(l,m) # for the total potential
        k = (sum(grav.an))/self.Ulm(l,m) # for the dynamical potential
        k_hs = (2*l + 1)*jn(l,np.pi) / (np.pi*jn(l-1,np.pi)) - 1
        #dk = (k - k_hs)/k_hs*100 # for the total potential
        self.dk = (k)/k_hs*100 # for the dynamical potential
        self.k = k
        self.Q = abs(np.absolute(k)/np.imag(k)) 
        return  
    
    def L(self):
        l = self.degree
        L = []
        [L.append( self.p()*self.cheby.d2Tn_x2(n) + self.q()*self.cheby.dTn_x(n) + self.r()*self.cheby.Tn(n) ) for n in self.n]

        # append the BC
        bc1 = []
        #[bc1.append( self.cheby.dTndx_bot(n) -l/self.x0*self.cheby.Tn_bot(n) ) for n in self.n]
        [bc1.append( self.cheby.Tn_bot(n) ) for n in self.n]
        #[bc1.append( self.cheby.dTndx_bot(n) -l/1e-10*self.cheby.Tn_bot(n) ) for n in self.n]
        bc2 = []
        [bc2.append( self.cheby.dTndx_top(n) + (l+1)/self.xr*self.cheby.Tn_top(n) ) for n in self.n]
        self.L = np.vstack( (np.array(L).T, bc1, bc2) ) # add the boundary conditions 
        return 
    
    
class gravity:
    # This is the gravity potential for ONE spherical harmonic
    def __init__(self, cheby, l=2, m=2, f=None,x0=1e-3,xr=3.14,Ulm=1):
        self.degree = l
        self.order = m
        self.f = f
        self.cheby = cheby
        self.n = np.arange(0,cheby.N)
        self.x0 = x0
        self.xr = xr
        self.Ulm = Ulm

        self.phi = []
        self.dphi = []
        self.d2phi = []
        self.an = []
        
    def p(self):
        return np.ones(len(self.cheby.xi))

    def q(self):
        return 2/self.cheby.xi

    def r(self):
        l = self.degree
        return 1 - l*(l+1)/self.cheby.xi**2

    def u(self):
        u = []
        [u.append(self.cheby.Tn(n)) for n in self.n]
        du = []
        [du.append(self.cheby.dTn_x(n)) for n in self.n]
        d2u = []
        [d2u.append(self.cheby.d2Tn_x2(n)) for n in self.n]
        return (np.dot(np.array(u).T, self.an), np.dot(np.array(du).T, self.an),np.dot(np.array(d2u).T, self.an))

    def solve(self):
        l = self.degree
        m = self.order
        #Bigf = np.concatenate( (self.f, [0, (2*l+1)/self.xr*self.Ulm]) ) # Add the third coefficient in the bc eqn. For the overall potential
        Bigf = np.concatenate( (self.f, [0, 0]) ) # Add the third coefficient in the bc eqn. For the dynamical potential
        #Bigf = np.concatenate( (self.f, [self.Ulm*(2*l+1)/np.pi*jn(l,self.x0)/jn(l-1,np.pi), (2*l+1)/self.xr*self.Ulm]) ) # Add the third coefficient in the bc eqn.
        self.L()
        self.an = self.cheby.lin_solve(self.L, Bigf)
        self.phi, self.dphi, self.d2phi = self.u()
        return  
    
    def L(self):
        l = self.degree
        L = []
        [L.append( self.p()*self.cheby.d2Tn_x2(n) + self.q()*self.cheby.dTn_x(n) + self.r()*self.cheby.Tn(n) ) for n in self.n]

        # append the BC
        bc1 = []
        #[bc1.append( self.cheby.dTndx_bot(n) -l/self.x0*self.cheby.Tn_bot(n) ) for n in self.n]
        [bc1.append( self.cheby.Tn_bot(n) ) for n in self.n]
        #[bc1.append( self.cheby.dTndx_bot(n) -l/1e-10*self.cheby.Tn_bot(n) ) for n in self.n]
        bc2 = []
        [bc2.append( self.cheby.dTndx_top(n) + (l+1)/self.xr*self.cheby.Tn_top(n) ) for n in self.n]
        self.L = np.vstack( (np.array(L).T, bc1, bc2) ) # add the boundary conditions 
        return 

class dynamical:
    # The coupled problem of dynamical tides including the non-perturbative Coriolis effect.
    def __init__(self, cheby,om, OM, Mp, ms,a, Rp, 
                 l=[2,4,6], m=2,tau=1e8,b=np.pi,
                 A=4.38,B=0,x0=1e-3,xr=3.14,G=6.67e-8,
                 tides='c', e=0):
        self.degree = l
        self.order = m
        self.OM = OM
        self.om = om
        self.a = a
        self.gravity_factor = G*ms/a
        self.Rp = Rp
        self.tau = tau
        self.rdip = 1
        self.ldamp = 2
        self.om2 = om +1j/tau
        self.G = G
        self.A = 1
        self.B = 0
        self.rhoc = A
        self.g = G*Mp/Rp**2
        self.tides = tides
        self.e = e

        self.cheby = cheby
        self.n = np.arange(0,cheby.N)
        self.N = cheby.N
        self.x0 = x0
        self.xr = xr

        self.psi = []
        self.dpsi = []
        self.d2psi = []
        self.phi = []
        self.phi_dyn = []
        self.dk = []
        self.k = []
        self.k_hs = []
        self.Q = []
    
    def rho(self):
        xi = self.cheby.xi
        return (self.A*np.sin(xi) + self.B*np.cos(xi))/xi
        #return (self.A-self.OM/2/np.pi/self.G)*jn(0,xi) + self.OM**2/2/np.pi/self.G*(1-jn(0,xi))

    def drho(self):
        xi = self.cheby.xi
        return ( np.cos(xi)*(self.A*xi-self.B) - np.sin(xi)*(self.A+self.B*xi) )/xi**2
        #return -(self.A-self.OM/2/np.pi/self.G)*jn(1,xi) + self.OM**2/2/np.pi/self.G*jn(1,xi) 


    def Ulm(self, l,m):
        """
        Numerical factor in the tidal forcing.

        kind:
            'c': conventional tides.
            'e': eccentricity tides.
            'o': obliquity tides.
        """
        
        if self.tides is 'c':
            if l >= 2:
                return self.gravity_factor*(self.Rp/self.a)**l *np.sqrt(4*np.pi*factorial(l-m)/(2*l+1)/factorial(l+m))*Plm(m,l,0)
            else:
                return 0

        elif self.tides is 'ee':
            if l >= 2:
                #return self.gravity_factor*(self.Rp/self.a)**l * 42/4*np.sqrt(2*np.pi/15)*self.e
                return self.gravity_factor*(self.Rp/self.a)**l * self.e *(l + 2*m + 1)*np.sqrt(4*np.pi*factorial(l-m)/(2*l+1)/factorial(l+m))*Plm(m,l,0)
            else: 
                return 0

    def f(self):
        xi = self.cheby.xi
        m = self.order
        om = self.om
        om2=self.om2
        inx = np.argwhere(xi>(1-self.rdip)*np.pi)[-1][0]
        tau = np.ones(len(xi))*1e20
        tau[:inx]=self.tau
        phit = []
        [ phit.append( self.Ulm(l,m)*(om**2-4*self.OM**2)/(4*np.pi*self.G*self.rhoc)*(2*l+1)/self.xr*jn(l,xi)/jn(l-1,self.xr) ) for l in self.degree ]
        phit = np.concatenate(phit)
        return phit 

    # For p,q,r and the outerboundary condition, I have two implementations: (1) CG coefficients and (2) recursive relations. The second is preferred. 
    def p(self):
        F = self.OM/self.om2
        xi = self.cheby.xi
        j0 = self.rho()
        j1 = -self.drho()
        m = self.order
        inx = np.argwhere(xi>(1-self.rdip)*np.pi)[-1][0]
        tau = np.ones(len(xi))*1e20
        tau[:inx]=self.tau
        pfunc = []
        pfunc2 = []
        for l in self.degree: 
            Q = self.Qlm

            # term from l-2
            # Define Clebsh-Gordan coefficients
            CC = float( (CG(2,0,l-2,0,l,0)*CG(2,0,l-2,m,l,m)).doit().evalf() )
            #pfunc2.append( -j0*8/3*(self.OM/self.om2)**2 * ((2*l-3)/(2*l+1))**0.5 * CC  )
            pfunc2.append( -4*F**2*j0*Q(-1+l)*Q(l)   )

            # term from l
            CC = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
            #pfunc2.append( j0 - 4/3*(self.OM/self.om2)**2 * (j0 + 2*j0*CC) )
            pfunc2.append( j0*(1 - 4*F**2*(Q(l)**2 + Q(1 + l)**2))  )

            # term from l+2
            CC = float( (CG(2,0,l+2,0,l,0)*CG(2,0,l+2,m,l,m)).doit().evalf() )
            #pfunc2.append( -j0*8/3*(self.OM/self.om2)**2 * ((2*l+5)/(2*l+1))**0.5 * CC  )
            pfunc2.append( -4*F**2*j0*Q(1 + l)*Q(2 + l) )

            pfunc.append(pfunc2)
            pfunc2 = []
        return pfunc

    def q(self):
        F = self.OM/self.om2
        xi = self.cheby.xi
        j0 = self.rho()
        j1 = -self.drho()
        m = self.order
        inx = np.argwhere(xi>(1-self.rdip)*np.pi)[-1][0]
        tau = np.ones(len(xi))*1e20
        tau[:inx]=self.tau
        qfunc = []
        qfunc2 = []
        for l in self.degree: 
            Q = self.Qlm

            # term from l-2
            CC = float( (CG(2,0,l-2,0,l,0)*CG(2,0,l-2,m,l,m)).doit().evalf() )
            #qfunc2.append( 8/3*(self.OM/self.om2)**2 * ((2*l-3)/(2*l+1))**0.5 * (j1+(2*l-3)*j0/xi) * CC )
            qfunc2.append( (4*F**2*((-3 + 2*l)*j0 + xi*j1)*Q(-1 + l)*Q(l))/xi  )

            # term from l
            CC1 = float( (CG(1,0,l-1,0,l,0)*CG(1,0,l-1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
            #qfunc2.append( 2*j0/xi - j1 - 8/3*(self.OM/self.om2)**2 * ( 3*j0/xi*(l**2-m**2)**0.5 * CC1 - j1/2 - j0/xi*(l-1) - (j1+(2*l+1)*j0/xi) * CC2 ) )
            qfunc2.append( (j0*(2 - 4*F**2 + 4*F**2*(1 + 2*l)*(-Q(l)**2 + Q(1 + l)**2)))/xi + j1*(-1 + 4*F**2*(Q(l)**2 + Q(1 + l)**2))  )

            # term from l+2
            CC1 = float( (CG(1,0,l+1,0,l,0)*CG(1,0,l+1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l+2,0,l,0)*CG(2,0,l+2,m,l,m)).doit().evalf() )
            #qfunc2.append( -8/3*(self.OM/self.om2)**2 * ((2*l+5)/(2*l+1))**0.5 * ( 3*j0/xi*((l+2)**2 - m**2)**0.5 * CC1 - (j1+(2*l+5)*j0/xi) * CC2 ) )
            qfunc2.append( (4*F**2*((3 - 2*l)*j0 + xi*j1)*Q(1 + l)*Q(2 + l))/xi  )

            qfunc.append(qfunc2)
            qfunc2 = []
        return qfunc

    def r(self):
        F = self.OM/self.om2
        xi = self.cheby.xi
        j0 = self.rho()
        j1 = -self.drho()
        m = self.order
        inx = np.argwhere(xi>(1-self.rdip)*np.pi)[-1][0]
        tau = np.ones(len(xi))*1e20
        tau[:inx]=self.tau
        rfunc = []
        rfunc2 = []
        for l in self.degree: 
            Q = self.Qlm

            # term from l-2
            CC = float( (CG(2,0,l-2,0,l,0)*CG(2,0,l-2,m,l,m)).doit().evalf() )
            #rfunc2.append( -8/3*(self.OM/self.om2)**2 * ((2*l-3)/(2*l+1))**0.5  * (l-2)/xi * (j1-(l-2)*j0/xi) * CC )
            rfunc2.append( -((4*F**2*(-2 + l)*(l*j0 + xi*j1)*Q(-1 + l)*Q(l))/xi**2)  )

            # term from l
            CC1 = float( (CG(1,0,l-1,0,l,0)*CG(1,0,l-1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
            #rfunc2.append(-l*(l+1)*j0/xi**2 +  2*m*self.OM/self.om2*j1/xi - 4*(self.OM/self.om2)**2 * ( j0/3/xi**2*(3*m**2+2*l**2+3*l) + l*j1/3/xi + 2/3*l/xi*(j1-l*j0/xi) * CC2 - (l**2-m**2)**0.5*(j1/xi+j0/xi**2) * CC1) +  (self.om2**2-4*self.OM**2)/(4*np.pi*self.G)*self.om/self.om2 ) 
            rfunc2.append( (2*F*xi*j1*(m + 2*F*(1 + l)*Q(l)**2 - 2*F*l*Q(1 + l)**2) + l*(1 + l)*j0*(-1 + 4*F**2*(Q(l)**2 + Q(1 + l)**2)))/xi**2  )
            
            # term from l+2
            CC1 = float( (CG(1,0,l+1,0,l,0)*CG(1,0,l+1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l+2,0,l,0)*CG(2,0,l+2,m,l,m)).doit().evalf() )
            #rfunc2.append( -4*(self.OM/self.om2)**2 * ((2*l+5)/(2*l+1))**0.5 * ( 2/3/xi*(l+2)*(j1-(l+2)*j0/xi) * CC2 - ((l+2)**2 - m**2)**0.5*(j1/xi+j0/xi**2) * CC1 ) )
            rfunc2.append( -((4*F**2*((11 + l*(4 + l))*j0 + (1 - l)*xi*j1)*Q(1 + l)*Q(2 + l))/xi**2)  )

            rfunc.append(rfunc2)
            rfunc2 = []
        return rfunc
    
    def Qlm(self,l):
        m = abs(self.order)
        if l>=m:
            return np.sqrt((l-m)*(l+m)/(2*l-1)/(2*l+1))
        else:
            return 0

    # Boundary condition for the flow at the outer boundary or a vanishing point in the interior.
    def flowBc(self,surface=True):
        F = self.OM/self.om2
        if surface:
            x = self.xr
        else:
            x = self.x0
        m = self.order
        OM = self.OM
        om = self.om
        om2 = self.om2
        rfunc = []
        rfunc2 = []
        qfunc = []
        qfunc2 = []
        for l in self.degree: 
            Q = self.Qlm

            CC = float( (CG(2,0,l-2,0,l,0)*CG(2,0,l-2,m,l,m)).doit().evalf() )
            #qfunc2.append( -8/3*OM**2/om2**2*np.sqrt( (2*l-3)/(2*l+1) )*CC )
            qfunc2.append( -4*F**2*Q(l-1)*Q(l) )

            CC = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
            #qfunc2.append( 1 - 4/3*OM**2/om2**2*(1+2*CC) )
            qfunc2.append( 1 - 4*F**2*(Q(l+1)**2+Q(l)**2) )

            CC = float( (CG(2,0,l+2,0,l,0)*CG(2,0,l+2,m,l,m)).doit().evalf() )
            #qfunc2.append( -8/3*OM**2/om2**2*np.sqrt((2*l+5)/(2*l+1))*CC )
            qfunc2.append( -4*F**2*Q(l+2)*Q(l+1) )

            CC = float( (CG(2,0,l-2,0,l,0)*CG(2,0,l-2,m,l,m)).doit().evalf() )
            #rfunc2.append( 8/3*OM**2/om2**2*np.sqrt((2*l-3)/(2*l+1)) *(l-2)/x*CC )
            rfunc2.append( 4*F**2/x*(l-2)*Q(l-1)*Q(l) )
            
            CC1 = float( (CG(1,0,l-1,0,l,0)*CG(1,0,l-1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
            #rfunc2.append( -2/x*m*OM/om2 - 4/x*OM**2/om2**2*(-2*l/3*CC2 - l/3 + (l**2-m**2)**(1/2)*CC1) )
            rfunc2.append( -2/x*m*F + 4*F**2/x*(l*Q(l+1)**2-(l+1)*Q(l)**2)  )
            
            CC1 = float( (CG(1,0,l+1,0,l,0)*CG(1,0,l+1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l+2,0,l,0)*CG(2,0,l+2,m,l,m)).doit().evalf() )
            #rfunc2.append( -4/x*OM**2/om2**2*((2*l+5)/(2*l+1))**(1/2)*( ((l+2)**2-m**2)**(1/2)*CC1 - 2*(l+2)/3*CC2 ) ) 
            rfunc2.append( -4*F**2*(l-1)*Q(l+2)*Q(l+1)/x ) 


            rfunc.append(rfunc2)
            qfunc.append(qfunc2)
            rfunc2 = []
            qfunc2 = []
        return [qfunc, rfunc]
    
    def centerBc(self):
        nbc = len(self.degree)
        def bcb(i):
            b = []
            [b.append( self.cheby.dTndx_bot(n) - self.degree[i]/self.x0*self.cheby.Tn_bot(n) ) for n in self.n]
            return np.array(b)

        cols = []
        [cols.append( np.concatenate([ (np.concatenate([np.zeros(self.N)]*col) if col is not 0 else np.array([])), bcb(col), (np.concatenate([np.zeros(self.N)]*(nbc-1-col)) if (nbc-1-col)  is not 0 else np.array([]) )  ]) ) for col in range(0,nbc) ]
        return np.array(cols)
    
    def coreBc(self):
        # This is an experiment
        nbc = len(self.degree)
        def bcb(i):
            b = []
            #[b.append( self.cheby.Tn_bot(n) ) for n in self.n]
            #[b.append( self.cheby.dTndx_bot(n) - self.degree[i]/1e-10*self.cheby.Tn_bot(n) ) for n in self.n]
            [b.append( self.cheby.dTndx_bot(n)  ) for n in self.n]
            return np.array(b)

        cols = []
        [cols.append( np.concatenate([ (np.concatenate([np.zeros(self.N)]*col) if col is not 0 else np.array([])), bcb(col), (np.concatenate([np.zeros(self.N)]*(nbc-1-col)) if (nbc-1-col)  is not 0 else np.array([]) )  ]) ) for col in range(0,nbc) ]
        return np.array(cols)

    def BigflowBc(self, kind='top'):
        # This needs a thorough check. I found a bug in the block indecis describing the coupling amout tides.
        if kind is 'top':
            flowBc = self.flowBc(surface=True) 
        elif kind is 'bot':
            flowBc = self.flowBc(surface=False) 

        def block(i,j):
            B = []
            if kind is 'top':
                [B.append( flowBc[0][i][j]*self.cheby.dTndx_top(n) + flowBc[1][i][j]*self.cheby.Tn_top(n) ) for n in self.n]
            elif kind is 'bot':
                [B.append( flowBc[0][i][j]*self.cheby.dTndx_bot(n) + flowBc[1][i][j]*self.cheby.Tn_bot(n) ) for n in self.n]

            return np.array(B).T
        nxblocks = len(self.degree)
        first_row = np.concatenate( [ block(0,1) , block(0,2) , np.concatenate([ np.zeros((self.N))] * (nxblocks-2) ) ] )
        last_row = np.concatenate( [ np.concatenate([ np.zeros((self.N))] * (nxblocks-2) ) , block(nxblocks-1,0) , block(nxblocks-1,1) ] )
        middle_rows = []
        [ middle_rows.append( np.concatenate([ (np.concatenate([np.zeros((self.N))] * col) if col is not 0 else []), block(col,0), block(col,1), block(col,2), (np.concatenate([np.zeros((self.N))] * (nxblocks-3-col)) if (nxblocks-3-col) is not 0 else [] ) ]) ) for col in range(0,nxblocks-2) ]
        return np.vstack([first_row,middle_rows,last_row])

    def BigL(self, kind='bvp'):
        xi = self.cheby.xi
        nxblocks = len(self.degree)
        PQR = [self.p(), self.q(), self.r()]
        def block(i,j):
            #i: position of l in self.degree
            #j: coupling. j=0:l-2, j=1:l, and j=2:l+2
            B = []
            [B.append( PQR[0][i][j]*self.cheby.d2Tn_x2(n) + PQR[1][i][j]*self.cheby.dTn_x(n) + PQR[2][i][j]*self.cheby.Tn(n) ) for n in self.n]
            return np.array(B).T
        first_col = np.concatenate( [ block(0,1) , block(1,0), np.concatenate([np.zeros((len(xi),self.N))] * (nxblocks-2))] )
        last_col = np.concatenate( [ np.concatenate([np.zeros((len(xi),self.N))] * (nxblocks-2) ), block(nxblocks-2,2), block(nxblocks-1,1) ])

        middle_cols = []
        [ middle_cols.append( np.concatenate([ (np.concatenate([np.zeros((len(xi),self.N))] * col) if col is not 0 else np.array([[]]*self.N).T), block(col,2), block(col+1,1), block(col+2,0), (np.concatenate([np.zeros((len(xi),self.N))] * (nxblocks-3-col)) if (nxblocks-3-col) is not 0 else np.array([[]]*self.N).T ) ]) ) for col in range(0,nxblocks-2) ]
        
        L = np.concatenate( [first_col,np.concatenate(middle_cols,axis=1),last_col],axis=1 )
       
        # append boundary conditions
        if kind is 'ivp':
            # NEEDS UPDATE
            self.BigL = np.vstack( (L, self.coupledBc(CLO,kind='lower'), self.coupledBc(CUP,kind='lower')) ) 
        elif kind == 'bvp':
            self.BigL = np.vstack( (L, self.centerBc(), self.BigflowBc(kind='top')) )  
        elif kind == 'bvp-core':
            self.BigL = np.vstack( (L, self.BigflowBc(kind='bot'), self.BigflowBc(kind='top')) )  
            #self.BigL = np.vstack( (L, self.coreBc(), self.BigflowBc(kind='top')) )  

    def solve(self, kind='bvp'):
        C3 = []
        m = self.order
        if kind is 'ivp':
            # NEEDS UPDATE
            return
        elif kind == 'bvp' or kind == 'bvp-core':
            # NOTE: the evanescent central region (core) does not change fbc=0.
            FBC = [] 
            for l in self.degree: 
                if l > 1:
                    FBC.append( self.Ulm(l,m)*(4*self.OM**2-self.om2**2)/self.g*(2*l+1)/self.xr*jn(l,self.xr)/jn(l-1,self.xr) )
                else:
                    FBC.append(0+0j)

            Bigf = np.concatenate( (self.f(), np.zeros(len(self.degree)), FBC ) )

        self.BigL(kind)
        
        self.an = self.cheby.lin_solve( self.BigL, Bigf)

        self.psi, self.dpsi, self.d2psi = self.u()
        
        # Solve the gravity 
        for i in range(0,len(self.degree)):
            l = self.degree[i]
            grav = gravity(self.cheby, l=l, m=m,f=self.psi[i],x0=self.x0,xr=self.xr,Ulm=self.Ulm(l,m) )
            grav.solve()
            self.phi.append(grav.phi-self.Ulm(l,m)*(self.cheby.xi/self.xr)**l)
            #self.phi_dyn.append(grav.phi-self.Ulm(l,m)*(2*l+1)/self.xr*jn(l,self.cheby.xi)/jn(l-1,self.xr) )
            self.phi_dyn.append(grav.phi )
            
            # Love number calculations
            #k = (sum(grav.an)-self.Ulm(l,m))/self.Ulm(l,m) # for the total potential
            k = (sum(grav.an))/self.Ulm(l,m) # for the dynamical potential
            k_hs = (2*l + 1)*jn(l,np.pi) / (np.pi*jn(l-1,np.pi)) - 1
            #dk = (k - k_hs)/k_hs*100 # for the total potential
            dk = (k)/k_hs*100 # for the dynamical potential
            self.dk.append(dk)
            self.k.append(k)
            self.k_hs.append(k_hs)
            self.Q.append( abs(np.absolute(k)/np.imag(k)) )

        return 

    
    def u(self):
        pol = []
        [pol.append(self.cheby.Tn(n)) for n in self.n]
        u = []
        [ u.append( np.dot(np.array(pol).T, a)) for a in np.split(self.an,len(self.degree)) ]
        
        pol = []
        [pol.append(self.cheby.dTn_x(n)) for n in self.n]
        du = []
        [ du.append( np.dot(np.array(pol).T, a)) for a in np.split(self.an,len(self.degree)) ]
        
        pol = []
        [pol.append(self.cheby.d2Tn_x2(n)) for n in self.n]
        d2u = []
        [ d2u.append( np.dot(np.array(pol).T, a)) for a in np.split(self.an,len(self.degree)) ]
        return [u,du,d2u]

 

class dynamicalLeadingDegree:
    # The coupled problem of dynamical tides approximated for leading order. The coupled terms are zero.
    # An approximation for the leading degree tide where the coupling is neglected
    def __init__(self, om, omd, OM, Gms_a, a, Rp, l=2, m=2,tau=1e10):
        self.degree = l
        self.order = m
        self.OM = OM
        self.om = om
        self.omd = omd
        self.a = a
        self.gravity_factor = Gms_a
        self.Rp = Rp
        self.tau = tau

    def Ulm(self, l,m):
        if l >= 2:
            return self.gravity_factor*(self.Rp/self.a)**l *np.sqrt(4*np.pi/(2*l+1)/factorial(l+m))*Plm(m,l,0)
        else:
            return 0

    def f(self, xi):
        m = self.order
        l = self.degree
        tau = self.tau
        om = self.om
        #return self.Ulm(l,m)*(self.om**2 - 4*self.OM**2)/self.omd**2*(2*l+1)/np.pi*jn(l,xi)/jn(l-1,np.pi) 
        return self.Ulm(l,m)/self.omd**2*(2*l+1)/np.pi*jn(l,xi)/jn(l-1,np.pi)*-1j*om/(1j*om-1/tau)*((1j*om-1/tau)**2 +4*self.OM**2) 

    def p(self, xi):
        j0 = jn(0,xi)
        j1 = jn(1,xi)
        m = self.order
        l = self.degree
        tau = self.tau

        # term from l
        CC = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
        #return j0 *(1- 4/3*(self.OM/self.om)**2 * (1 + 2*CC)) 
        return j0 *(1+ 4/3*(self.OM/(1j*self.om-1/tau))**2 * (1 + 2*CC)) 

    def q(self, xi):
        j0 = jn(0,xi)
        j1 = jn(1,xi)
        m = self.order
        l = self.degree

        # term from l
        CC = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
        #return 2*j0/xi - j1 - 8/3*(self.OM/self.om)**2 * ( - j1/2 - j0/xi*(l-1) - (j1+(2*l+1)*j0/xi) * CC ) 
        return 2*j0/xi - j1 + 8/3*(self.OM/(1j*self.om-1/self.tau))**2 * ( - j1/2 - j0/xi*(l-1) - (j1+(2*l+1)*j0/xi) * CC ) 

    def r(self, xi):
        j0 = jn(0,xi)
        j1 = jn(1,xi)
        m = self.order
        l = self.degree

        # term from l
        CC = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
        #return  -l*(l+1)*j0/xi**2 -  2*m*self.OM/self.om*j1/xi - 4*(self.OM/self.om)**2 * ( j0/3/xi**2*(3*m**2+2*l**2+3*l) + l*j1/3/xi + 2/3*l/xi*(j1-l*j0/xi) * CC ) 
        return  -l*(l+1)*j0/xi**2 -  1j*2*m*self.OM/(1j*self.om-1/self.tau)*j1/xi + 4*(self.OM/(1j*self.om-1/self.tau))**2 * ( j0/3/xi**2*(3*m**2+2*l**2+3*l) + l*j1/3/xi + 2/3*l/xi*(j1-l*j0/xi) * CC ) 

