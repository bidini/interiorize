# Set of ODES defining the tidal response of an index-2 polytrope.
# Solve for the full tide (dynamic+hydrostatic) with the Cowlng approximation
#
# To do:
# - Check the coefficients using mathematica.
# - Check the calculation of the love numbers.
# - Test the hydrostatic case and compare with the analytical solution.
# - Compare the result with the analytical dynamical solution when there is no bottom boundary.
# - Check the consistency of the sign in omega and m.
# - Code the three tidal forcings for eccentricity and obliquity tides.
# - Maintain the tidal forcing for conventional tides.
# - Make the uniform density an input parameter.
# - Check the solution in a shell.
# - Create plots for the flow (black background with spheres in color).
# - Check if Chebyshev coefficients work well when one of the boundaries is not at the center.
# - Check if we need radius normalization.
# - Take the evaluation of Bigf out of solve()
# - Change the bottom/top boundaries to x1 and x2

import numpy as np
from scipy.special import spherical_jn as jn
from scipy.special import lpmv as Plm
from scipy.special import factorial
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
    def __init__(self, cheby, om, OM, Mp, ms, a, Rp, 
                l=[2,4,6], m=2, tau=1e8, b=np.pi, A=4.38,
                B=0, x0=1e-3, xr=3.14, G=6.67e-8, k2_hydro=1.5):
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

        self.cheby = cheby
        self.n = np.arange(0,cheby.N)
        self.N = cheby.N
        self.x0 = x0
        self.xr = xr

        self.psi = []
        self.dpsi = []
        self.d2psi = []
        self.flow = []
        self.phi = []
        self.phi_dyn = []
        self.dk = []
        self.k = []
        self.Q = []
        self.k2_hydro = k2_hydro
    
    def rho(self):
        return 1.33*np.ones(len(self.cheby.xi)) 

    # amplitude of conventional tidal forcing
    def Ulm(self, l, m):
        # define cases for eccentricity and obliquity and non tidally locked.
        if l >= 2:
            return self.gravity_factor*(self.Rp/self.a)**l *np.sqrt(4*np.pi*factorial(l-m)/(2*l+1)/factorial(l+m))*Plm(m,l,0)
        else:
            return 0
    
    # forcing
    def f(self):
        xi = self.cheby.xi
        phit = []
        [ phit.append( np.zeros(len(xi)) ) for l in self.degree ]
        phit = np.concatenate(phit)
        return phit 

    # For p,q,r and the outerboundary condition, using recursive relations.  
    # psi''
    def p(self):
        F = self.OM/self.om2
        xi = self.cheby.xi
        m = self.order
        pfunc = []
        pfunc2 = []
        for l in self.degree: 
            Q = self.Qlm

            # term from l-2
            pfunc2.append( -4*F**2*Q(-1+l)*Q(l)   )

            # term from l
            pfunc2.append( 1 - 4*F**2*(Q(l)**2 + Q(1 + l)**2)  )

            # term from l+2
            pfunc2.append( -4*F**2*Q(1 + l)*Q(2 + l) )

            pfunc.append(pfunc2)
            pfunc2 = []
        return pfunc
    
    # psi' 
    def q(self):
        F = self.OM/self.om2
        xi = self.cheby.xi
        m = self.order
        qfunc = []
        qfunc2 = []
        for l in self.degree: 
            Q = self.Qlm

            # term from l-2
            qfunc2.append( (4*F**2*(-3 + 2*l)*Q(-1 + l)*Q(l))/xi  )

            # term from l
            qfunc2.append( (2 - 4*F**2 + 4*F**2*(1 + 2*l)*(-Q(l)**2 + Q(1 + l)**2))/xi )

            # term from l+2
            qfunc2.append( (4*F**2*(3 - 2*l)*Q(1 + l)*Q(2 + l))/xi  )

            qfunc.append(qfunc2)
            qfunc2 = []
        return qfunc
    
    # psi
    def r(self):
        F = self.OM/self.om2
        xi = self.cheby.xi
        m = self.order
        rfunc = []
        rfunc2 = []
        for l in self.degree: 
            Q = self.Qlm

            # term from l-2
            rfunc2.append( -((4*F**2*(-2 + l)*l*Q(-1 + l)*Q(l))/xi**2)  )

            # term from l
            rfunc2.append( (l*(1 + l)*(-1 + 4*F**2*(Q(l)**2 + Q(1 + l)**2)))/xi**2  )
            
            # term from l+2
            rfunc2.append( -((4*F**2*(11 + l*(4 + l))*Q(1 + l)*Q(2 + l))/xi**2)  )

            rfunc.append(rfunc2)
            rfunc2 = []
        return rfunc
    
    # Recursive coefficients
    def Qlm(self,l):
        m = abs(self.order)
        if l>=m:
            return np.sqrt((l-m)*(l+m)/(2*l-1)/(2*l+1))
        else:
            return 0

    # Radial flow boundary condition: dp = 0 at r=R and vr = 0 in the interior.
    # The BC right-hand side is added later; this is just vr.
    def flowBc(self,surface=True):

        # evaluate x for the radial coefficients
        if surface:
            x = self.xr
        else:
            x = self.x0
        
        F = self.OM/self.om2
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

            # term from l-2
            qfunc2.append( -4*F**2*Q(l-1)*Q(l) )

            # term from l
            qfunc2.append( 1 - 4*F**2*(Q(l+1)**2+Q(l)**2) )

            # term from l+2
            qfunc2.append( -4*F**2*Q(l+2)*Q(l+1) )

            # term from l-2
            rfunc2.append( 4*F**2/x*(l-2)*Q(l-1)*Q(l) )
            
            # term from l
            #rfunc2.append( -2/x*m*F + 4*F**2/x*(l*Q(l+1)**2-(l+1)*Q(l)**2)  # modified)
            rfunc2.append( 2/x*m*F + 4*F**2/x*(l*Q(l+1)**2-(l+1)*Q(l)**2) + (4*OM**2-om**2)/self.g )
            
            # term from l+2
            #rfunc2.append( -4*F**2*(l-1)*Q(l+2)*Q(l+1)/x ) # I found a typo here. 
            rfunc2.append( -4*F**2*(-l-3)*Q(l+2)*Q(l+1)/x )  

            rfunc.append(rfunc2)
            qfunc.append(qfunc2)
            rfunc2 = []
            qfunc2 = []
        return [qfunc, rfunc]
    
    ## ----------------------------------------------------------------------------------------------
    # Start building matrices
    # The method is based on calculating coefficients that multiple analytical expressions for the 
    # derivatives of Chebyshev polynomials.

    def centerBc(self):
        """
        Inner boundary condition (r=0): make psi well-behaved near the center without specifying the flow.
        This BC has no radial terms.
        """

        nbc = len(self.degree)
        def bcb(i):
            b = []
            [b.append( self.cheby.dTndx_bot(n) - self.degree[i]/self.x0*self.cheby.Tn_bot(n) ) for n in self.n]
            return np.array(b)

        cols = []
        [cols.append( np.concatenate([ (np.concatenate([np.zeros(self.N)]*col) if col is not 0 else np.array([])), bcb(col), (np.concatenate([np.zeros(self.N)]*(nbc-1-col)) if (nbc-1-col)  is not 0 else np.array([]) )  ]) ) for col in range(0,nbc) ]

        return np.array(cols)
    
    def BigflowBc(self, kind='top'):
        """
        Build the boundary condition matrix
        """

        if kind is 'top':
            # surface bc
            flowBc = self.flowBc(surface=True) 
        elif kind is 'bot':
            # bottom bc
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
        """
        Build the problem matrix
        """

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
            # IVP NEEDS UPDATE
            self.BigL = np.vstack( (L, self.coupledBc(CLO,kind='lower'), self.coupledBc(CUP,kind='lower')) ) 

        elif kind == 'bvp':
            # Flow covers the entire body, from center to surface
            self.BigL = np.vstack( (L, self.centerBc(), self.BigflowBc(kind='top')) )  

        elif kind == 'bvp-core':
            # Flow is restricted to a shell (i.e., global ocean)
            self.BigL = np.vstack( (L, self.BigflowBc(kind='bot'), self.BigflowBc(kind='top')) )  

    # End building matrices
    ## ----------------------------------------------------------------------------
    
    # Solve the problem
    def solve(self, kind='bvp'):
        m = self.order
        if kind is 'ivp':
            # IVP NEEDS UPDATE 
            return
        elif kind == 'bvp' or kind == 'bvp-core':
            # NOTE: the evanescent central region (core) does not change fbc=0.
            FBC = [] 
            for l in self.degree: 
                # I only add l=2 because we only care about k2; others can be added later.
                if l == 2:
                    # add the forcing (right-hand side) in the outer boundary condition (dp=0)
                    FBC.append( self.Ulm(l,m)*(4*self.OM**2-self.om2**2)/self.g*(1+self.k2_hydro) )
                else:
                    FBC.append(0+0j)

            Bigf = np.concatenate( (self.f(), np.zeros(len(self.degree)), FBC ) )

        self.BigL(kind)
        
        self.an = self.cheby.lin_solve( self.BigL, Bigf)

        self.psi, self.dpsi, self.d2psi = self.u()
    
        self.get_flow()
        
        # Solve the love number 
        for i in range(0,len(self.degree)):
            l = self.degree[i]
            
            # gravity is obtained only at the surface
            phi = self.flow[i][-1]*self.g + self.psi[i][-1] 

            self.phi.append(grav.phi-self.Ulm(l,m)*(self.cheby.xi/self.xr)**l)
            self.phi_dyn.append(grav.phi )
            
            # Calculate the tidal Love numbers
            k = (sum(grav.an))/self.Ulm(l,m) 
            k_hs = self.k2_hydro 
            dk = (k)/k_hs*100 
            self.dk.append(dk)
            self.k.append(k)

            self.Q.append( abs(np.absolute(k)/np.imag(k)) )

        return 
    
    # Build the flow (radial displacement)
    def get_flow(self):
        psi = self.psi
        dpsi = self.dpsi
        d2psi = self.d2psi
        m = self.order
        OM = self.OM
        om = self.om
        F = OM/om
        pdb.set_trace()
        
        for i in range(0,len(self.degree)):
            Q = self.Qlm
            l = self.degree[i]
            
            # From a Mathematica template. Check attached file 'SH_projection.nb'
            c1 = 2*m + 4*F**2*((-1-l)*Q(l)**2+l*Q(1+l)**2) 
            c2 = 4*F**2*(-3-l)*Q(1+l)*Q(2+l) 
            c3 = 4*F**2*(-2+l)*Q(-1+l)*Q(l) 
            c4 = 1 - 4*F**2*(Q(l)**2+Q(1+l)**2) 
            c5 = -4*F**2*Q(1+l)*Q(2+l)
            c6 = -4*F**2*Q(-1+l)*Q(l)

            if l == self.degree[0]: 
                d = (c1*psi[i] + c2*psi[i+1] + c4*dpsi[i] + c5*dpsi[i+1])/(4*OM**2-om**2) 
            elif l == self.degree[-1]:
                d = (c1*psi[i] + c3*psi[i-1] + c4*dpsi[i] + c6*dpsi[i-1])/(4*OM**2-om**2) 
            else:
                d = (c1*psi[i] + c2*psi[i+1] + c3*psi[i-1] + c4*dpsi[i] + c5*dpsi[i+1] + c6*dpsi[i-1])/(4*OM**2-om**2) 

            self.flow.append(d)

        return

    # Build the Chebyshev solution 
    def u(self):
        pol = []
        [pol.append(self.cheby.Tn(n)) for n in self.n]
        u = []
        [ u.append( np.dot(np.array(pol).T, a)) for a in np.split(self.an, len(self.degree)) ]
        
        pol = []
        [pol.append(self.cheby.dTn_x(n)) for n in self.n]
        du = []
        [ du.append( np.dot(np.array(pol).T, a)) for a in np.split(self.an, len(self.degree)) ]
        
        pol = []
        [pol.append(self.cheby.d2Tn_x2(n)) for n in self.n]
        d2u = []
        [ d2u.append( np.dot(np.array(pol).T, a)) for a in np.split(self.an, len(self.degree)) ]

        return [u, du, d2u]
