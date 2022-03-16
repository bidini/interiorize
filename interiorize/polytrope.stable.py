# Set of ODES defining the tidal response of an index-2 polytrope.

import numpy as np
from scipy.special import spherical_jn as jn
from scipy.special import lpmv as Plm
from scipy.special import factorial
from sympy.physics.quantum.cg import CG
import pdb

class norotation:
    # This is the flow potential without Coriolis. Vorontsov's dynamical tide.
    def __init__(self,om,omd,l=2):
        self.om = om
        self.omd = omd
        self.degree = l

    def f(self,xi):
        l = self.degree
        phit = 5*(self.om/self.omd)**l * jn(l,xi)
        return  phit + phit*1e-5j

    def p(self,xi):
        return jn(0,xi)

    def q(self,xi):
        return ( 2*jn(0,xi)/xi - jn(1,xi) )

    def r(self,xi):
        l = self.degree
        return - jn(0,xi) * l*(l+1) / xi**2
    
class gravity:
    # This is the gravity potential
    def __init__(self, l=2):
        self.degree = l

    def p(self,xi):
        return np.ones(len(xi))

    def q(self,xi):
        return 2/xi

    def r(self,xi):
        l = self.degree
        return 1 - l*(l+1)/xi**2
    
class static:
    # Solve the static gravity (benchmark)
    def f(self,xi,l=2):
        return -(xi/np.pi)**l 

    def p(self,xi,l=2):
        return np.ones(len(xi))

    def q(self,xi,l=2):
        return 2/xi 

    def r(self,xi,l=2):
        return -l*(l+1)/xi**2 + 1

class dynamical:
    # The coupled problem of dynamical tides including the non-perturbative Coriolis effect.
    def __init__(self, om, omd, OM, Gms_a, a, Rp, l=[2,4,6], m=2,tau=1e8):
        self.degree = l
        self.order = m
        self.OM = OM
        self.om = om
        self.omd = omd
        self.a = a
        self.gravity_factor = Gms_a
        self.Rp = Rp
        self.tau = tau
        self.rdip = 1
        self.ldamp = 1

    def Ulm(self, l,m):
        if l >= 2:
            return self.gravity_factor*(self.Rp/self.a)**l *np.sqrt(4*np.pi*factorial(l-m)/(2*l+1)/factorial(l+m))*Plm(m,l,0)
        else:
            return 0

    def f(self, xi):
        m = self.order
        om = self.om
        inx = np.argwhere(xi>(1-self.rdip)*np.pi)[-1][0]
        tau = np.ones(len(xi))*1e20
        tau[:inx]=self.tau
        phit = []
        [ phit.append( self.Ulm(l,m)/self.omd**2*(2*l+1)/np.pi*jn(l,xi)/jn(l-1,np.pi)*-1j*om/(1j*om-1/(1e20 if l <=self.ldamp else self.tau))*((1j*om-1/(1e20 if l<=self.ldamp else self.tau))**2 +4*self.OM**2) ) for l in self.degree ]
        phit = np.concatenate(phit)
        return phit 

    def p(self, xi):
        j0 = jn(0,xi)
        j1 = jn(1,xi)
        m = self.order
        inx = np.argwhere(xi>(1-self.rdip)*np.pi)[-1][0]
        tau = np.ones(len(xi))*1e20
        tau[:inx]=self.tau
        pfunc = []
        pfunc2 = []
        for l in self.degree: 
            tau = 1e20 if l <= self.ldamp else self.tau
            # term from l-2
            # Define Clebsh-Gordan coefficients
            CC = float( (CG(2,0,l-2,0,l,0)*CG(2,0,l-2,m,l,m)).doit().evalf() )
            pfunc2.append( j0*8/3*(self.OM/(1j*self.om-1/tau))**2 * ((2*l-3)/(2*l+1))**0.5 * CC  )
            #pfunc2.append( -j0*8/3*(self.OM/self.om)**2 * ((2*l-3)/(2*l+1))**0.5 * CC  )

            # term from l
            CC = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
            pfunc2.append( j0 + 4/3*(self.OM/(1j*self.om-1/tau))**2 * (j0 + 2*j0*CC) )
            #pfunc2.append( j0 - 4/3*(self.OM/self.om)**2 * (j0 + 2*j0*CC) )

            # term from l+2
            CC = float( (CG(2,0,l+2,0,l,0)*CG(2,0,l+2,m,l,m)).doit().evalf() )
            pfunc2.append( j0*8/3*(self.OM/(1j*self.om-1/tau))**2 * ((2*l+5)/(2*l+1))**0.5 * CC  )
            #pfunc2.append( -j0*8/3*(self.OM/self.om)**2 * ((2*l+5)/(2*l+1))**0.5 * CC  )

            pfunc.append(pfunc2)
            pfunc2 = []
        return pfunc

    def q(self, xi):
        j0 = jn(0,xi)
        j1 = jn(1,xi)
        m = self.order
        inx = np.argwhere(xi>(1-self.rdip)*np.pi)[-1][0]
        tau = np.ones(len(xi))*1e20
        tau[:inx]=self.tau
        qfunc = []
        qfunc2 = []
        for l in self.degree: 
            tau = 1e20 if l <= self.ldamp else self.tau
            # term from l-2
            CC = float( (CG(2,0,l-2,0,l,0)*CG(2,0,l-2,m,l,m)).doit().evalf() )
            qfunc2.append( -8/3*(self.OM/(1j*self.om-1/tau))**2 * ((2*l-3)/(2*l+1))**0.5 * (j1+(2*l-3)*j0/xi) * CC )
            #qfunc2.append( 8/3*(self.OM/self.om)**2 * ((2*l-3)/(2*l+1))**0.5 * (j1+(2*l-3)*j0/xi) * CC )

            # term from l
            CC1 = float( (CG(1,0,l-1,0,l,0)*CG(1,0,l-1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
            qfunc2.append( 2*j0/xi - j1 + 8/3*(self.OM/(1j*self.om-1/tau))**2 * ( 3*j0/xi*(l**2-m**2)**0.5 * CC1 - j1/2 - j0/xi*(l-1) - (j1+(2*l+1)*j0/xi) * CC2 ) )
            #qfunc2.append( 2*j0/xi - j1 - 8/3*(self.OM/self.om)**2 * ( 3*j0/xi*(l**2-m**2)**0.5 * CC1 - j1/2 - j0/xi*(l-1) - (j1+(2*l+1)*j0/xi) * CC2 ) )

            # term from l+2
            CC1 = float( (CG(1,0,l+1,0,l,0)*CG(1,0,l+1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l+2,0,l,0)*CG(2,0,l+2,m,l,m)).doit().evalf() )
            qfunc2.append( 8/3*(self.OM/(1j*self.om-1/tau))**2 * ((2*l+5)/(2*l+1))**0.5 * ( 3*j0/xi*((l+2)**2 - m**2)**0.5 * CC1 - (j1+(2*l+5)*j0/xi) * CC2 ) )
            #qfunc2.append( -8/3*(self.OM/self.om)**2 * ((2*l+5)/(2*l+1))**0.5 * ( 3*j0/xi*((l+2)**2 - m**2)**0.5 * CC1 - (j1+(2*l+5)*j0/xi) * CC2 ) )

            qfunc.append(qfunc2)
            qfunc2 = []
        return qfunc

    def r(self, xi):
        j0 = jn(0,xi)
        j1 = jn(1,xi)
        m = self.order
        inx = np.argwhere(xi>(1-self.rdip)*np.pi)[-1][0]
        tau = np.ones(len(xi))*1e20
        tau[:inx]=self.tau
        rfunc = []
        rfunc2 = []
        for l in self.degree: 
            tau = 1e20 if l <= self.ldamp else self.tau
            # term from l-2
            CC = float( (CG(2,0,l-2,0,l,0)*CG(2,0,l-2,m,l,m)).doit().evalf() )
            rfunc2.append( 8/3*(self.OM/(1j*self.om-1/tau))**2 * ((2*l-3)/(2*l+1))**0.5  * (l-2)/xi * (j1-(l-2)*j0/xi) * CC )
            #rfunc2.append( -8/3*(self.OM/self.om)**2 * ((2*l-3)/(2*l+1))**0.5  * (l-2)/xi * (j1-(l-2)*j0/xi) * CC )

            # term from l
            CC1 = float( (CG(1,0,l-1,0,l,0)*CG(1,0,l-1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
            rfunc2.append(  -l*(l+1)*j0/xi**2 - 1j*2*m*self.OM/(1j*self.om-1/tau)*j1/xi + 4*(self.OM/(1j*self.om-1/tau))**2 * ( j0/3/xi**2*(3*m**2+2*l**2+3*l) + l*j1/3/xi + 2/3*l/xi*(j1-l*j0/xi) * CC2 - (l**2-m**2)**0.5*(j1/xi+j0/xi**2) * CC1 )) 
            #rfunc2.append(  -l*(l+1)*j0/xi**2 -  2*m*self.OM/self.om*j1/xi - 4*(self.OM/self.om)**2 * ( j0/3/xi**2*(3*m**2+2*l**2+3*l) + l*j1/3/xi + 2/3*l/xi*(j1-l*j0/xi) * CC2 - (l**2-m**2)**0.5*(j1/xi+j0/xi**2) * CC1 )) 
            
            # term from l+2
            CC1 = float( (CG(1,0,l+1,0,l,0)*CG(1,0,l+1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l+2,0,l,0)*CG(2,0,l+2,m,l,m)).doit().evalf() )
            rfunc2.append( 4*(self.OM/(1j*self.om-1/tau))**2 * ((2*l+5)/(2*l+1))**0.5 * ( 2/3/xi*(l+2)*(j1-(l+2)*j0/xi) * CC2 - ((l+2)**2 - m**2)**0.5*(j1/xi+j0/xi**2) * CC1 ) )
            #rfunc2.append( -4*(self.OM/self.om)**2 * ((2*l+5)/(2*l+1))**0.5 * ( 2/3/xi*(l+2)*(j1-(l+2)*j0/xi) * CC2 - ((l+2)**2 - m**2)**0.5*(j1/xi+j0/xi**2) * CC1 ) )


            rfunc.append(rfunc2)
            rfunc2 = []
        return rfunc

    def bc(self):
        m = self.order
        OM = self.OM
        om = self.om
        tau = self.tau
        rfunc = []
        rfunc2 = []
        qfunc = []
        qfunc2 = []
        for l in self.degree: 
            tau = 1e20 if l <= self.ldamp else self.tau
            CC = float( (CG(2,0,l-2,0,l,0)*CG(2,0,l-2,m,l,m)).doit().evalf() )
            #rfunc2.append( 8/3*OM**2/om**2*((2*l-3)/(2*l+1))**(1/2) *(l-2)/np.pi*CC )
            rfunc2.append( -8/3*OM**2/(1j*om-1/tau)**2*((2*l-3)/(2*l+1))**(1/2) *(l-2)/np.pi*CC )
            
            CC1 = float( (CG(1,0,l-1,0,l,0)*CG(1,0,l-1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
            #rfunc2.append( 2/np.pi*m*OM/om - 4*OM**2/om**2*(-2*l/3/np.pi*CC2 - l/3/np.pi + 1/np.pi*(l**2-m**2)**(1/2)*CC1) )
            rfunc2.append( 1j*2/np.pi*m*OM/(1j*om-1/tau) + 4*OM**2/(1j*om-1/tau)**2*(-2*l/3/np.pi*CC2 - l/3/np.pi + 1/np.pi*(l**2-m**2)**(1/2)*CC1) )
            
            CC1 = float( (CG(1,0,l+1,0,l,0)*CG(1,0,l+1,m,l,m)).doit().evalf() )
            CC2 = float( (CG(2,0,l+2,0,l,0)*CG(2,0,l+2,m,l,m)).doit().evalf() )
            #rfunc2.append( -4*OM**2/om**2*((2*l+5)/(2*l+1))**(1/2)*( ((l+2)**2-m**2)**(1/2)/np.pi*CC1 - 2*(l+2)/3/np.pi*CC2 ) ) 
            rfunc2.append( 4*OM**2/(1j*om-1/tau)**2*((2*l+5)/(2*l+1))**(1/2)*( ((l+2)**2-m**2)**(1/2)/np.pi*CC1 - 2*(l+2)/3/np.pi*CC2 ) ) 

            CC = float( (CG(2,0,l-2,0,l,0)*CG(2,0,l-2,m,l,m)).doit().evalf() )
            #qfunc2.append( -8/3*OM**2/om**2*np.sqrt( (2*l-3)/(2*l+1) )*CC )
            qfunc2.append( 8/3*OM**2/(1j*om-1/tau)**2*np.sqrt( (2*l-3)/(2*l+1) )*CC )

            CC = float( (CG(2,0,l,0,l,0)*CG(2,0,l,m,l,m)).doit().evalf() )
            #qfunc2.append( 1 - 4/3*OM**2/om**2*(1+2*CC) )
            qfunc2.append( 1 + 4/3*OM**2/(1j*om-1/tau)**2*(1+2*CC) )

            CC = float( (CG(2,0,l+2,0,l,0)*CG(2,0,l+2,m,l,m)).doit().evalf() )
            #qfunc2.append( -8/3*OM**2/om**2*((2*l+5)/(2*l+1))**(1/2)*CC )
            qfunc2.append( 8/3*OM**2/(1j*om-1/tau)**2*((2*l+5)/(2*l+1))**(1/2)*CC )

            rfunc.append(rfunc2)
            qfunc.append(qfunc2)
            rfunc2 = []
            qfunc2 = []
        return [qfunc, rfunc]

    def fbc(self,G,k,rhoc):
        m = self.order
        tau = self.tau
        om = self.om
        f = []
        for l in self.degree: 
            if l >=2:
                tau = 1e20 if l <= self.ldamp else self.tau
                rhoi = self.Ulm(l,m)*k**2/(4*np.pi*G)*(2*l+1)/np.pi*jn(l,np.pi)/jn(l-1,np.pi)
                drhoi = self.Ulm(l,m)*k**2/(4*np.pi*G)*(2*l+1)/np.pi*(jn(l-1,np.pi)-(l+1)/np.pi*jn(l,np.pi))/jn(l-1,np.pi)
                #f.append( -(self.om**2 - 4*self.OM**2)/2/k**2*rhoi*(rhoc*jn(1,np.pi)-drhoi) ) 
                f.append( 1j*om/(1j*om-1/tau)*((1j*om-1/tau)**2 + 4*self.OM**2)/2/k**2*rhoi*(rhoc*jn(1,np.pi)-drhoi) ) 
            else:
                f.append(0.)

        return np.array(f)
 

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

