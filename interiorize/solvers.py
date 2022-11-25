# Compilation of elements required for a numerical solution of a Sturmn-Luoville problem using chebyshev polynomials.
# The ODE is in the form:
#
# p(x) y''(x) + q(x) y'(x) + r(x) y(x) = f(x)
#
# with boundary conditions in the form:
#
# c1 y'(a) + c2 y(a) = c3 
#
# Based on Boyd 2001, Chapter 6.
#
# Benjamin Idini April 2020

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.special import spherical_jn as jn
from scipy.special import lpmv as Plm
from sympy.physics.quantum.cg import CG
from math import factorial
from scipy.linalg import block_diag
from scipy.integrate import odeint, ode
import pdb


#################################################################################
class cheby:

    def __init__(self, npoints=10, loend=0., upend=np.pi, direction=1, extrema=False):
        # npoints:  number of Chebyshev polynomials
        # loend:    lower boundary
        # upend:    upper boundary
        
        self.extrema = extrema
        self.N = npoints
        self.loend = loend
        self.upend = upend
        
        # Compute the interior collocation points (spectral grid)
        #self.x = np.cos(np.pi*np.arange(0, npoints)/(npoints-1))    # x in (-1,1). Chebyshev grid for boundary bordering 
        #self.x = np.cos(np.pi*(2*np.arange(1, npoints+1)-1)/2/(npoints))    # x in (-1,1). Gauss-Chebyshev 
        #self.x = np.cos(np.pi*np.arange(0, npoints)/(npoints-1))    # x in (-1,1). Gauss-Lobatto (includes end points) 
        # NOTE: x may contain the boundary of [-1,1] but then the ChebyPols derivatives need to be evaluated using limits.
        
        if extrema:
            self.x = np.cos(np.pi*np.arange(0, npoints)/(npoints-1))[::direction]
        else:
            self.x = np.cos(np.pi*np.arange(1, npoints-1)/(npoints-1))[::direction]

        self.xi = (loend+upend)/2 + self.x*(upend-loend)/2    # (loend,upend): Collocation points in the domain of the S-L problem
        # NOTE: xi is only used to evaluate the differential equation along radius in a planet's interior

        self.dxi_dx = (upend-loend)/2    # chain rule factor
        self.T = np.arccos(self.x) # Collocation points in the trigonometric domain (just of practical use)
        self.a = [] # Chebyshev coefficients

        # Why are collocation points so important in this problem? results change a lot.

    # Define the Chebyshev polynomials and derivatives (N-2)
    # n:    order of polynomial
    # t:    vector within domain where to evaluate polynomials
    def Tn(self, n,t=np.array(0)):
        if t.all() ==0:
            t = self.T
        return np.cos(n*t) 

    def dTn_x(self, n):
        if self.extrema:
        # evalute the derivatives using limits
            t = self.T[1:-1]
            pd = n*np.sin(n*t)/np.sin(t)
            pd = np.append(pd,n**2)
            pd = np.insert(pd,0,-(-1)**n * n**2)
        else:
            t = self.T
            pd = n*np.sin(n*t)/np.sin(t)

        return pd/self.dxi_dx 

    def d2Tn_x2(self, n):
        if self.extrema:
            t = self.T[1:-1]
            pd = (-n**2*np.cos(n*t)/np.sin(t)**2 + n*np.cos(t)*np.sin(n*t)/np.sin(t)**3)
            pd = np.append(pd, 1/3*n**2*(n**2 -1) )
            pd = np.insert(pd,0, -1/3*n**2*(-1)**n + 1/3*n**4*(-1)**n )
        else:
            t = self.T
            pd = (-n**2*np.cos(n*t)/np.sin(t)**2 + n*np.cos(t)*np.sin(n*t)/np.sin(t)**3)

        return pd/self.dxi_dx**2 #multiply or divide

    # (Note: Only Tn has an argument where the domain of the chevyshev changes, t. This is required to evaluate the solution at points that are
    # different from the collocation points when evaluating the quality of the result using FD.)

    # Define the boundary condition
    # Tn(-1) = (-1)**n ; Tn(1) = 1
    # dTn_x(-1) = n**2 (-1)**(n+1) ; dTn_x(1) = n**2
    def Tn_top(self,n):
        return 1
    def Tn_bot(self,n):
        return (-1)**n
    def dTndx_top(self,n):
        return n**2 / self.dxi_dx
    def dTndx_bot(self,n):
        return (-1)**(n+1) * n**2 / self.dxi_dx
    
    def lin_solve(self, L, f):
        return np.linalg.solve( L, f)

    # Obtain a solution by adding the series of Chebyshevs up to N
    def u(self, t):
        u = []
        [u.append(self.Tn(n,t)) for n in np.arange(0,self.N)]
        du = []
        [du.append(self.dTn_x(n)) for n in np.arange(0,self.N)]
        d2u = []
        [d2u.append(self.d2Tn_x2(n)) for n in np.arange(0,self.N)]
        return (np.dot(np.array(u).T, self.a), np.dot(np.array(du).T, self.a),np.dot(np.array(d2u).T, self.a))

    def coupledU(self, t):
        pol = []
        [pol.append(self.Tn(n,t)) for n in np.arange(0,self.N)]
        u = []
        [ u.append( np.dot(np.array(pol).T, a)) for a in np.split(self.a,len(self.a)//self.N) ]
        
        pol = []
        [pol.append(self.dTn_x(n)) for n in np.arange(0,self.N)]
        du = []
        [ du.append( np.dot(np.array(pol).T, a)) for a in np.split(self.a,len(self.a)//self.N) ]
        
        pol = []
        [pol.append(self.d2Tn_x2(n)) for n in np.arange(0,self.N)]
        d2u = []
        [ d2u.append( np.dot(np.array(pol).T, a)) for a in np.split(self.a,len(self.a)//self.N) ]
        return [u,du,d2u]

    # Define the square matrix and column vector to solve the linear system L*m = f
    # p,q,r:    (n,) functions for the coefficients of the S-L problem
    # clo:      (3,) list/array [c1,c2,c3] of coefficients in the lower bc
    # cup:      (3,) list/array [c1,c2,c3] of coefficients in the upper bc
    def L(self, p, q, r, clo, cup, kind='bvp'):
        xi = self.xi
        L = []
        [L.append( p(xi)*self.d2Tn_x2(n) + q(xi)*self.dTn_x(n) + r(xi)*self.Tn(n,self.T) ) for n in np.arange(0,self.N)]
        if kind is 'bvp':
            return np.vstack( (np.array(L).T, self.bc(clo,kind='lower'), self.bc(cup,kind='upper')) ) # add the boundary conditions 
        elif kind is 'ivp':
            return np.vstack( (np.array(L).T, self.bc(clo,kind='lower'), self.bc(cup,kind='lower')) ) # add the boundary conditions 
    
    def updatedL(self, P, Q, R, CLO, CUP, kind='bvp', BC=None):
        # work in progress
        Lpsi = self.coupledL(P, Q, R, CLO, CUP, kind=kind, BC=BC)
        
        # gravity potential matrix
        Lphi = [] 
        #[Lphi.append( np.ones(len(xi))*self.d2Tn_x2(n) + 2/xi*self.dTn_x(n) + (1-)*self.Tn(n,self.T) ) for n in np.arange(0,self.N)]
        return

    def coupledL(self, P, Q, R, CLO, CUP, kind='bvp', BC=None,BCc=None):
        xi = self.xi
        nxblocks = len(CLO)
        PQR = [P(xi), Q(xi), R(xi)]
        def block(i,j):
            B = []
            [B.append( PQR[0][i][j]*self.d2Tn_x2(n) + PQR[1][i][j]*self.dTn_x(n) + PQR[2][i][j]*self.Tn(n,self.T) ) for n in np.arange(0,self.N)]
            return np.array(B).T
        first_col = np.concatenate( [ block(0,1) , block(1,0), np.concatenate([np.zeros((len(xi),self.N))] * (nxblocks-2))] )
        last_col = np.concatenate( [ np.concatenate([np.zeros((len(xi),self.N))] * (nxblocks-2) ), block(nxblocks-2,2), block(nxblocks-1,1) ])

        middle_cols = []
        [ middle_cols.append( np.concatenate([ (np.concatenate([np.zeros((len(xi),self.N))] * col) if col is not 0 else np.array([[]]*self.N).T), block(col,2), block(col+1,1), block(col+2,0), (np.concatenate([np.zeros((len(xi),self.N))] * (nxblocks-3-col)) if (nxblocks-3-col) is not 0 else np.array([[]]*self.N).T ) ]) ) for col in range(0,nxblocks-2) ]
        
        L = np.concatenate( [first_col,np.concatenate(middle_cols,axis=1),last_col],axis=1 )
        
        if kind is 'bvp':
            return np.vstack( (L, self.coupledBc(CLO,kind='lower'), self.coupledBc(CUP,kind='upper')) ) # add the boundary conditions 
        elif kind is 'ivp':
            return np.vstack( (L, self.coupledBc(CLO,kind='lower'), self.coupledBc(CUP,kind='lower')) ) # add the boundary conditions 
        elif kind == 'bvp-coupled':
            return np.vstack( (L, self.coupledBc(CLO,kind='lower'), self.mechBc(BC)) ) # add the boundary conditions 
        elif kind == 'bvp-withcore':
            return np.vstack( (L, self.mechBc(BCc,kind='bot'), self.mechBc(BC,kind='top')) ) # add the boundary conditions 

#################################################################################

class bessel:
    # Use Spherical bessel functions of the first kind as basis functions.
    # This basis behaves as the function we need: the boundary conditions are satisfied imposing a0 = a1 = 0
    # Never tested
    def __init__(self, npoints=10, loend=0., upend=np.pi):
        # npoints:  number of functions
        # loend:    lower boundary
        # upend:    upper boundary

        self.N = npoints
        self.x0 = loend
        self.x1 = upend
        
        # Compute equidistant collocation points
        self.xi = np.pi*np.arange(1,npoints-2)/(npoints-2)    
        self.a = [] # Chebyshev coefficients

    def djn(self,n,xi):
        return n/xi*jn(n,xi)-jn(n+1,xi)

    def d2jn(self,n,xi):
        return 1/4*(-jn(n,xi)+(2*n-1)/xi*jn(n-1,xi)) + (3-2*xi**2)/4/xi**2*jn(n,xi) + 1/2/xi*jn(n+1,xi) + 1/4*jn(n+2,xi)-1/2/xi*jn(n-1,xi)

    def L(self, p, q, r, clo, cup, kind='bvp'):
        xi = self.xi
        L = []
        [L.append( p(xi)*self.d2jn(n,xi) + q(xi)*self.djn(n,xi) + r(xi)*jn(n,xi) ) for n in np.arange(1,self.N)]
        if kind is 'bvp':
            return np.vstack( (np.array(L).T, self.bc(clo,kind='lower'), self.bc(cup,kind='upper')) ) # add the boundary conditions 
        elif kind is 'ivp':
            return np.vstack( (np.array(L).T, self.bc(clo,kind='lower'), self.bc(cup,kind='lower')) ) # add the boundary conditions 
        else:
            return np.vstack( (np.array(L).T, self.bc(clo,kind='lower')) )
    
    def bc(self, c, kind='lower'):
        bc = []
        if kind is 'lower':
            [bc.append(  c[1]*jn(n,self.x0) + c[0]*self.djn(n,self.x0) ) for n in np.arange(1,self.N)]
        elif kind is 'upper':
            [bc.append( c[1]*jn(n,self.x1) + c[0]*self.djn(n,self.x1) ) for n in np.arange(1,self.N)]
        return bc
    
    def u(self, xi):
        u = []
        xi = np.concatenate((xi,np.array([self.x1])))
        [u.append(jn(n,xi)) for n in np.arange(1,self.N)]
        du = []
        [du.append(self.djn(n,xi)) for n in np.arange(1,self.N)]
        d2u = []
        [d2u.append(self.d2jn(n,xi)) for n in np.arange(1,self.N)]
        return (np.dot(np.array(u).T, self.a), np.dot(np.array(du).T, self.a),np.dot(np.array(d2u).T, self.a))
    
    def solve(self, p, q, r, f, clo, cup, kind='bvp'):
        f_bc = np.concatenate( (f(self.xi), [clo[2], cup[2]]) ) # Add the third coefficient in the bc eqn.
        L = self.L(p,q,r,clo,cup,kind)
        self.a = np.linalg.solve( L, f_bc)
        return self.a 

#################################################################################
class integrator:
    # A time integrator to solve the radial equation for the potential velocity
    def __init__(self, npoints=10, loend=0., upend=np.pi):
        self.N = npoints
        self.x0 = loend
        self.x1 = upend

    def solve(self,p, q, r, f):

        def dpsi_dx(x,psi):
            return [ psi[1], (f(x)-r(x)*psi[0]-q(x)*psi[1])/p(x) ]
        
        def jac(x,psi):
            return [ [0,1] , [-r(x)/p(x),-q(x)/p(x)] ]
        S = ode(dpsi_dx,jac).set_integrator('vode',method='bdf')
        S.set_initial_value([0,0],self.x0)
        dx=(self.x1-self.x0)/(self.N-1)
        sol = [0]
        while S.successful() and S.t < self.x1:
            sol.append(S.integrate(S.t+dx)[0])

        return np.array(sol)


