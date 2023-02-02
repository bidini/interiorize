# Solutions to the Poincare problem.
#
# Incompressible water ocean with Coriolis and a rigid bottom core.
#
# Ben Idini, Dec 2022.

"""
To do:
- Check if we need radius normalization. We do. Check if we can get rid of it.
- Check the solution in a shell. It works. Is it right? Plot the flow and search for the well-known attractors.

Low priority:

- Code the three tidal forcings for eccentricity and obliquity tides. Check Chen+ (Nimmo paper)
- Add the boundary values (analytical) to the chebyshev solution
- Check if Chebyshev coefficients work well when one of the boundaries is not at the center. Maintain the tidal forcing for conventional tides.
- Create plots for the flow (black background with spheres in color).
- Fix the typos found here also in polytrope.py

"""


import numpy as np
from scipy.special import spherical_jn as jn
from scipy.special import lpmv as Plm
from scipy.special import factorial
import pdb
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

class dynamical:
    def __init__(self, cheby, om, OM, Mp, ms, a, Rp, 
                Rc=0, l=[2,4,6], m=2, tau=1e8, b=np.pi, rho=1, rhoc=0,
                x1=1e-3, x2=3.14, G=6.67e-8, N2=0,
                 forcing='homogeneous', e=0, tides='c'):
        """
        The coupled problem of dynamical tides including the non-perturbative Coriolis effect.

            om: tidal frequency
            OM: spin frequency
            Mp: mass of the body
            ms: mass of the companion
            a:  semi-major axis of companion
            Rp: radius of the body
            N2: Brunt-Vaisala frequency squared to indicate stratification
        """

        self.degree = l
        self.order = m
        self.OM = OM
        self.om = om
        self.a = a
        self.gravity_factor = G*ms/a
        self.Rp = Rp
        self.Rc = Rc
        self.tau = tau
        self.om2 = om + 1j/tau
        self.G = G
        self.rho = rho
        self.rhoc = rhoc
        self.forcing = forcing
        self.N2 = N2
        self.e = e
        self.tides = tides

        # gravity at the surface of an ocean with a rigid core
        self.g = 4/3*np.pi*G*Rp*rho*(1 + (rhoc/rho-1)*(Rc/Rp)**3 ) 
        
        self.cheby = cheby
        self.n = np.arange(0,cheby.N)
        self.N = cheby.N
        self.x1 = x1
        self.x2 = x2

        self.psi = []
        self.dpsi = []
        self.d2psi = []
        self.flow = []
        self.flow_sum = []
        self.phi = []
        self.phi_dyn = []
        self.dk = []
        self.k = []
        self.Q = []

    def k_hydro(self, l):
        """
        The hydrostatic Love number in a differentiated body with an ocean and a perfectly rigid core (from my notes).
        """
        return 3/(2*(l-1) + (2*l+1)*(self.rhoc/self.rho-1)*(self.Rc/self.Rp)**3) 
    
    def Ulm(self, l, m):
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
            if (l >= 2) and (l<=6):
                #return self.gravity_factor*(self.Rp/self.a)**l * 42/4*np.sqrt(2*np.pi/15)*self.e
                return self.gravity_factor*(self.Rp/self.a)**l * self.e *(l + 2*m + 1)*np.sqrt(4*np.pi*factorial(l-m)/(2*l+1)/factorial(l+m))*Plm(m,l,0)
            else: 
                return 0


        elif self.tides is 'o':
            return
    
    ## ----------------------------------------------------------------------------------------------
    # Define coefficients for p,q,r,f and the outerboundary condition, using recursive relations.  
    
    def p(self):
        """
        psi''
        """

        F = self.OM/self.om2
        xi = self.cheby.xi
        m = self.order
        pfunc = []
        pfunc2 = []
        for l in self.degree: 
            Q = self.Qlm

            # term from l-2 (c9)
            pfunc2.append( -4*F**2*Q(-1+l)*Q(l)   )

            # term from l (c7)
            pfunc2.append( 1 - 4*F**2*(Q(l)**2 + Q(1 + l)**2)  )

            # term from l+2 (c8)
            pfunc2.append( -4*F**2*Q(1 + l)*Q(2 + l) )

            pfunc.append(pfunc2)
            pfunc2 = []

        return pfunc
    
    def q(self):
        """
        psi'
        """

        F = self.OM/self.om2
        xi = self.cheby.xi
        m = self.order
        qfunc = []
        qfunc2 = []
        for l in self.degree: 
            Q = self.Qlm

            # term from l-2 (c6)
            qfunc2.append( (4*F**2*(-3 + 2*l)*Q(-1 + l)*Q(l))/xi  )

            # term from l (c4)
            qfunc2.append( (2 - 4*F**2 + 4*F**2*(1 + 2*l)*(-Q(l)**2 + Q(1 + l)**2))/xi )

            # term from l+2 (c5)
            #qfunc2.append( (4*F**2*(3 - 2*l)*Q(1 + l)*Q(2 + l))/xi  ) # typo
            qfunc2.append( (-4*F**2*(5 + 2*l)*Q(1 + l)*Q(2 + l))/xi  )

            qfunc.append(qfunc2)
            qfunc2 = []

        return qfunc
    
    def r(self):
        """
        psi
        """

        F = self.OM/self.om2
        xi = self.cheby.xi
        m = self.order
        rfunc = []
        rfunc2 = []
        for l in self.degree: 
            Q = self.Qlm

            # term from l-2 (c3)
            rfunc2.append( -4*F**2*(-2 + l)*l*Q(-1 + l)*Q(l)/xi**2  )

            # term from l (c1)
            rfunc2.append( (l*(1 + l)*(-1 + 4*F**2*(Q(l)**2 + Q(1 + l)**2)))/xi**2  )
            
            # term from l+2 c(2)
            #rfunc2.append( -((4*F**2*(11 + l*(4 + l))*Q(1 + l)*Q(2 + l))/xi**2)  ) #typo
            rfunc2.append( -4*F**2*(3 + l*(4 + l))*Q(1 + l)*Q(2 + l)/xi**2  )

            rfunc.append(rfunc2)
            rfunc2 = []

        return rfunc
    
    def f(self, forcing='homogeneous'):
        """
        Right-hand side in the Poincare equation.

        type:   'homogeneous' for the homogeneous Poincare problem.
                'forced' for the salinity-forced problem.
        """

        xi = self.cheby.xi
        phit = []

        if forcing is 'homogeneous':
            print('Solving the homogeneous Poincare problem')
            [ phit.append( np.zeros(len(xi)) ) for l in self.degree ]

        elif forcing is 'forced':
            print('Solving the salinity-forced Poincare problem')
            # Save the homogeneous flow, previously computed
            self.flow0 = self.flow

            Q = self.Qlm
            F = self.OM/self.om2
            m = self.order

            for i in np.arange(len(self.degree)):
                l = self.degree[i]
                disp = self.flow0[i]
                d_disp = np.gradient(disp, xi)

                c1 = self.N2*(2 - 4*F**2 + 2*F*m - 4*F**2*l*Q(l)**2 + 4*F**2*(1+l)*Q(1+l)**2) / xi 
                c2 = -self.N2*4*F**2*(2+l)*Q(1+l)*Q(2+l) / xi 
                c3 = self.N2*4*F**2*(-1+l)*Q(-1+l)*Q(l)/ xi
                c4 = self.N2*(1 - 4*F**2*(Q(l)**2 + Q(1+l)**2))
                c5 = -self.N2*4*F**2*Q(1+l)*Q(2+l)
                c6 = -self.N2*4*F**2*Q(-1+l)*Q(l)

                if l == self.degree[0]: 
                    rhs = (c1*disp[i] + c2*disp[i+1] + c4*d_disp[i] + c5*d_disp[i+1])
                elif l == self.degree[-1]:
                    rhs = (c1*disp[i] + c3*disp[i-1] + c4*d_disp[i] + c6*d_disp[i-1]) 
                else:
                    rhs = (c1*disp[i] + c2*disp[i+1] + c3*disp[i-1] + c4*d_disp[i] + c5*d_disp[i+1] + c6*d_disp[i-1]) 

                # consider the renormalization self.Rp/np.pi
                # The normalization comes from ( 1/r or d/d_r ) = PI/(Rp*x).
                rhs = rhs*self.Rp/np.pi 

                phit.append( rhs )

        phit = np.concatenate(phit)
        return phit 
    
    def flowBc(self, surface=True):
        """
        Radial flow boundary condition: dp = 0 at r=R and vr = 0 in the interior.
        The BC right-hand side is added later; this is just vr.
        """

        # evaluate x for the radial coefficients
        if surface:
            x = self.x2
        else:
            x = self.x1
        
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

            # term from l-2 (c6)
            qfunc2.append( -4*F**2*Q(l-1)*Q(l) )

            # term from l (c4)
            qfunc2.append( 1 - 4*F**2*(Q(l+1)**2+Q(l)**2) )

            # term from l+2 (c5)
            qfunc2.append( -4*F**2*Q(l+2)*Q(l+1) )

            # term from l-2 (c3)
            rfunc2.append( 4*F**2/x*(l-2)*Q(l-1)*Q(l) )
            
            # term from l (c1)
            # fix to dissipation applied
            rfunc2.append( (-2/x*m*F + 4*F**2/x*(l*Q(l+1)**2-(l+1)*Q(l)**2)) + surface*(4*OM**2-om2**2)*om/om2/self.g*self.Rp/np.pi )
            
            # term from l+2 (c2)
            #rfunc2.append( -4*F**2*(l-1)*Q(l+2)*Q(l+1)/x ) # I found a typo here. 
            rfunc2.append( 4*F**2*(-l-3)*Q(l+2)*Q(l+1)/x )  

            rfunc.append(rfunc2)
            qfunc.append(qfunc2)
            rfunc2 = []
            qfunc2 = []

        return [qfunc, rfunc]
    
    def Qlm(self,l):
        """ 
        Recursivity coefficient.
        """

        m = abs(self.order)
        if l>=m:
            return np.sqrt((l-m)*(l+m)/(2*l-1)/(2*l+1))
        else:
            return 0

    ## ----------------------------------------------------------------------------------------------
    # Start building matrices
    # The method is based on calculating coefficients that multiple analytical expressions for the 
    # derivatives of Chebyshev polynomials.

    def centerBc(self):
        """
        Inner boundary condition (r=0): make psi well-behaved near the center without specifying the flow.
        This BC has no radial terms.
        """
        Q = self.Qlm
        nbc = len(self.degree)
        def bcb(i):
            b = []
            #[b.append( self.cheby.dTndx_bot(n) - self.degree[i]/self.x1*self.cheby.Tn_bot(n) ) for n in self.n]
            [b.append( self.cheby.dTndx_bot(n) ) for n in self.n]
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
        first_row = np.concatenate( [ block(0, 1) , block(0, 2) , np.concatenate([ np.zeros((self.N))] * (nxblocks-2) ) ] )
        last_row = np.concatenate( [ np.concatenate([ np.zeros((self.N))] * (nxblocks-2) ) , block(nxblocks-1, 0) , block(nxblocks-1, 1) ] )
        middle_rows = []
        [ middle_rows.append( np.concatenate([ (np.concatenate([np.zeros((self.N))] * col) if col is not 0 else []), block(col, 0), block(col, 1), block(col, 2), (np.concatenate([np.zeros((self.N))] * (nxblocks-3-col)) if (nxblocks-3-col) is not 0 else [] ) ]) ) for col in range(0, nxblocks-2) ]

        return np.vstack([first_row, middle_rows, last_row])
    
    def get_BigL(self, kind='bvp'):
        """
        Build the problem left-hand side matrix
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

        first_col = np.concatenate( [ block(0, 1) , block(1, 0), np.concatenate([np.zeros((len(xi), self.N))] * (nxblocks-2))] )
        last_col = np.concatenate( [ np.concatenate([np.zeros((len(xi), self.N))] * (nxblocks-2) ), block(nxblocks-2, 2), block(nxblocks-1, 1) ])

        middle_cols = []
        [ middle_cols.append( np.concatenate([ (np.concatenate([np.zeros((len(xi), self.N))] * col) if col is not 0 else np.array([[]]*self.N).T), block(col, 2), block(col+1, 1), block(col+2, 0), (np.concatenate([np.zeros((len(xi), self.N))] * (nxblocks-3-col)) if (nxblocks-3-col) is not 0 else np.array([[]]*self.N).T ) ]) ) for col in range(0, nxblocks-2) ]
        
        L = np.concatenate( [first_col, np.concatenate(middle_cols, axis=1), last_col], axis=1 )
       
        # append boundary conditions at the end of the matrix
        if kind is 'ivp':
            # IVP NEEDS UPDATE
            self.BigL = np.vstack( (L, self.coupledBc(CLO, kind='lower'), self.coupledBc(CUP, kind='lower')) ) 

        elif kind == 'bvp':
            # Flow covers the entire body, from center to surface
            self.BigL = np.vstack( (L, self.centerBc(), self.BigflowBc(kind='top')) )  

        elif kind == 'bvp-core':
            # Flow is restricted to a shell (i.e., global ocean)
            self.BigL = np.vstack( (L, self.BigflowBc(kind='bot'), self.BigflowBc(kind='top')) )  
    
    def get_Bigf(self, kind='bvp'):
        """
        Build the problem right-hand side matrix
        """

        m = self.order
        if kind is 'ivp':
            # IVP NEEDS UPDATE 
            return
        
        elif kind == 'bvp' or kind == 'bvp-core':
            # NOTE: both the evanescent central region (core) and central regularity have fbc=0.
            FBCtop = [] 
            for l in self.degree: 
                # add the forcing (right-hand side) in the outer boundary condition (dp=0)
                # Added a fix to dissipation.
                FBCtop.append( self.Ulm(l,m)*(1 + self.k_hydro(l))*(4*self.OM**2-self.om2**2)*self.om/self.om2/self.g*self.Rp/np.pi )
            
            # Concatenate the forcing and right-hand side of inner and outer boundary conditions.
            self.Bigf = np.concatenate( (self.f(forcing=self.forcing), np.zeros(len(self.degree)), FBCtop ) )

    # End building matrices
    ## ----------------------------------------------------------------------------
    
    def solve(self, kind='bvp'):
        """
        Solve the problem
        """

        self.get_Bigf(kind)

        self.get_BigL(kind)
        
        self.an = self.cheby.lin_solve( self.BigL, self.Bigf)

        self.psi, self.dpsi, self.d2psi = self.u()

        self.get_love()

        self.get_flow()

        return 

    # Evaluate love numbers
    def get_love(self):
        """
        Solve for the love number 
        """
        m = self.order
        self.dk = []
        self.k = []
        self.Q = []

        for i in range(0,len(self.degree)):
            l = self.degree[i]
            
            # gravity is obtained only at the surface
            indSurface = np.argmax(self.cheby.xi)
            
            dk = self.psi[i][indSurface]/(self.k_hydro(l)*self.Ulm(l,m))*(1 - self.g*(2*l+1)/(4*np.pi*self.G*self.rho*self.Rp))**(-1)*100

            k = self.k_hydro(l)*(1+dk/100)

            self.dk.append(dk)
            self.k.append(k)

            self.Q.append( abs(np.absolute(k)/np.imag(k)) )

            """
            sanity check

            5/2*self.Rp*self.om*(2*self.OM+self.om)/2/self.g*2.5*self.Ulm(l,m)
            dk = -5/2*self.psi[i][indSurface]/2.5/self.Ulm(l,m)*100
            """

        return

    # Build the radial displacement 
    def get_flow(self):
        psi = self.psi
        dpsi = self.dpsi
        d2psi = self.d2psi
        m = self.order
        OM = self.OM
        om = self.om
        om2 = self.om2
        F = OM/om2

        self.flow = []

        for i in range(0,len(self.degree)):
            Q = self.Qlm
            l = self.degree[i]
            r = self.cheby.xi
            
            # From a Mathematica template. Check attached file 'SH_projection.nb'
            c1 = -2*F*m + 4*F**2*((-1-l)*Q(l)**2+l*Q(1+l)**2) 
            c2 = 4*F**2*(-3-l)*Q(1+l)*Q(2+l) 
            c3 = 4*F**2*(-2+l)*Q(-1+l)*Q(l) 
            c4 = 1 - 4*F**2*(Q(l)**2+Q(1+l)**2) 
            c5 = -4*F**2*Q(1+l)*Q(2+l)
            c6 = -4*F**2*Q(-1+l)*Q(l)
            
            # total tidal flow in the inner region, including hydrostatic
            # fix to dissipation applied.
            if l == self.degree[0]: 
                d = (c1*psi[i]/r + c2*psi[i+1]/r + c4*dpsi[i] + c5*dpsi[i+1])/(4*OM**2-om2**2)*om2/om 
            elif l == self.degree[-1]:
                d = (c1*psi[i]/r + c3*psi[i-1]/r + c4*dpsi[i] + c6*dpsi[i-1])/(4*OM**2-om2**2)*om2/om 
            else:
                d = (c1*psi[i]/r + c2*psi[i+1]/r + c3*psi[i-1]/r + c4*dpsi[i] + c5*dpsi[i+1] + c6*dpsi[i-1])/(4*OM**2-om2**2)*om2/om 
           
            # denormalize by radius
            d = d/self.Rp*np.pi

            # flow at the boundaries
            if self.cheby.extrema is False:
                indSurface = np.argmax(self.cheby.xi)
                indInnerBC = np.argmin(self.cheby.xi)

                # surface
                r = self.cheby.xi[indSurface]

                # dynamical part of the tidal displacement. NOTE: I need the full part
                d2 = self.psi[i][indSurface]*( 4*np.pi*self.G*self.rho*self.Rp/(2*l+1) - self.g)**(-1)
                '''
                sanity check with surface gravity:

                '''
 
                # inner boundary
                d1 = 0

                # add d1 and d2 to the d vector
                #d = np.insert(d, indInnerBC, d1)
                #d = np.insert(d, indSurface, d2)

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
