# Solutions to internal gravity waves in a rotating shell.
#
# Incompressible water ocean with Coriolis and a rigid bottom core.
#
# Ben Idini, Feb 2022.

import numpy as np
from scipy.special import spherical_jn as jn
from scipy.special import lpmv as Plm
from scipy.special import factorial
import pdb
import matplotlib.pyplot as plt

class internal_phi:
    '''
    Solve the internal gravity gravity 
    '''

    def __init__(self, cheby, dyn):
        self.cheby = cheby # The chebyshev solver object
        self.dyn = dyn     # The dynamical problem object
        self.f = []

        self.phi = []
        self.dphi = []
        self.d2phi = []
        self.an  = []

    def p(self):
        return np.ones(len(self.cheby.xi))

    def q(self):
        return 2/self.cheby.xi 

    def r(self, i):
        l = self.dyn.degree[i]
        return 1 - l*(l+1)/self.cheby.xi**2

    def get_L(self, i):
        L = []
        [L.append( self.p()*self.cheby.d2Tn_x2(n) + self.q()*self.cheby.dTn_x(n) + self.r(i)*self.cheby.Tn(n) ) for n in self.n]
        return L

    def u(self):
        u = []
        [u.append(self.cheby.Tn(n)) for n in self.dyn.n]
        du = []
        [du.append(self.cheby.dTn_x(n)) for n in self.dyn.n]
        d2u = []
        [d2u.append(self.cheby.d2Tn_x2(n)) for n in self.dyn.n]
        return (np.dot(np.array(u).T, self.an), np.dot(np.array(du).T, self.an),np.dot(np.array(d2u).T, self.an))

    def solve(self):

        for i in range(len(self.dyn.degree)):
            L = self.get_L(i)
            f = np.concatenate( [self.f, 0 ,0] )
            
            # boundary conditions (they will require some thinking)
            bc1 = []
            [bc1.append( self.cheby.Tn_bot(n) ) for n in self.n]

            bc2 = []
            [bc2.append( self.cheby.dTndx_top(n) + (l+1)/self.x0*self.cheby.Tn_top(n) ) for n in self.n]

            L = np.concatenate( [np.array(L).T, bc1, bc2] )

            self.an.append( self.cheby.lin_solve( L, f, sparse=True) )
            phi, dphi, d2phi = self.u()
            self.phi.append(phi)

        return



class dynamical:
    def __init__(self, cheby, om, OM, Mp, ms, a, Rp, 
                Rc=0, l=[2,3,4], m=2, tau=1e8, b=np.pi, rho=1, rhoc=0,
                x1=1e-3, x2=3.14, G=6.67e-8,
                e=0, tides='c', N2=0):
        """
        The coupled problem of dynamical tides including the non-perturbative Coriolis effect.

            om: tidal frequency
            OM: spin frequency
            Mp: mass of the body
            ms: mass of the companion
            a:  semi-major axis of companion
            Rp: radius of the body
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
        if tau==0:
            self.om2 = self.om
        else:
            self.om2 = om + 1j/tau
        self.G = G
        self.rho = rho
        self.rhoc = rhoc
        self.e = e
        self.N2 = N2
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
        self.y1 = []
        self.dy1 = []
        self.d2y1 = []
        self.y2 = []
        self.y3 = []

        self.phi = []
        self.flow = []
        self.flow_sum = []
        self.phi = []
        self.phi_dyn = []
        self.dk = []
        self.k = []
        self.Q = []
        self.E = []

    def k_hydro(self, l):
        """
        The hydrostatic Love number in a differentiated body with a constant density ocean and a perfectly rigid core (from my notes).
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
        
        if self.tides == 'c':
            if l >= 2:

                return self.gravity_factor*(self.Rp/self.a)**l *np.sqrt(4*np.pi*factorial(l-m)/(2*l+1)/factorial(l+m))*Plm(m,l,0)

            else:

                return 0

        elif self.tides == 'ee':
            # Derived in notes.
            if (l >= 2) and (l<=8):
                # The numerical error is roughly 1e-13. Forcing higher than l=8 falls within the error.
                return self.gravity_factor*(self.Rp/self.a)**l * self.e *(l + 2*m + 1)*np.sqrt(4*np.pi*factorial(l-m)/(2*l+1)/factorial(l+m))*Plm(m,l,0)

            else: 

                return 0

        elif self.tides == 'o':

            return
    
    ## ----------------------------------------------------------------------------------------------
    # Define the coefficients in the equations.   

    def eqn1_y1(self, l):
        """
        momentum equation component Y combined with continuity.
        """
        x = self.cheby.xi
        m = self.order
        om = self.om
        om2 = self.om2
        OM = self.OM
        N2 = self.N2

        q = m/l/(l+1)*x*OM
        r = (om*om2 - N2)/(2*om) + 2*m*OM/l/(l+1)

        return [0,0,0], [0,q,r], [0,0,0]

    def eqn1_y3(self, l):
        m = self.order
        q = self.Qlm
        return [0,0,1j*q(l)*(l-1)*self.OM], [0,0,0], [0,0,-1j*q(l+1)*(l+2)*self.OM]

    def eqn1_psi(self, l):
        k = np.pi/self.Rp # x = kr
        return [0,0,0], [0,k/2/self.om,0], [0,0,0]

    def eqn2_y1(self, l):
        q = self.Qlm
        x = self.cheby.xi
        OM = self.OM

        q0 = q(l)/l**2*x*OM 
        r0 = -q(l)/l**2*(l-2)*OM
        q2 = q(l+1)/(l+1)**2*x*OM
        r2 = q(l+1)/(l+1)**2*(l+3)*OM
        return [0,q0,r0], [0,0,0], [0,q2,r2]
    
    def eqn2_y3(self, l):
        return [0,0,0], [0,0,1j*(self.order*self.OM/l/(l+1) - self.om2/2)], [0,0,0]

    def eqn3_y1(self, l):
        x = self.cheby.xi
        m = self.order
        om = self.om
        om2 = self.om2
        OM = self.OM
        N2 = self.N2

        p = 1j*(om2/2 - m*OM/l/(l+1))/l/(l+1)*x**2
        q = 1j*(om2/2 - m*OM/l/(l+1))/l/(l+1)*4*x 
        r = 1j*N2/2/om + 1j*(om2/2 - m*OM/l/(l+1))*(2/l/(l+1)-1)
        return [0,0,0], [p,q,r], [0,0,0]
    
    def eqn3_y3(self, l):
        q = self.Qlm
        x = self.cheby.xi
        OM = self.OM

        q0 = q(l)*(l-1)/l*x*OM
        r0 = -q(l)*(l-1)**2/l*OM
        q2 = q(l+1)*(l+2)/(l+1)*x*OM
        r2 = q(l+1)*(l+2)**2/(l+1)*OM
        return [0,q0,r0], [0,0,0], [0,q2,r2]

    def eqn_zeros(self, l):
        return [0,0,0], [0,0,0], [0,0,0]
    
    def f(self):
        """
        Right-hand side in the system of equations.

        """

        phit = []

        [ phit.append( np.zeros(len(self.cheby.xi)) ) for l in self.degree ]

        phit = np.concatenate(phit)

        return np.concatenate([phit]*3) 
    
    def Qlm(self, l):
        """ 
        Recursivity coefficient associated to spherical harmonic coupling.
        """

        m = self.order
        if l>=abs(m):

            return np.sqrt((l-m)*(l+m)/(2*l-1)/(2*l+1))

        else:

            return 0

    ## ----------------------------------------------------------------------------------------------
    # Start building matrices
    # The method is based on calculating coefficients that multiple analytical expressions for the 
    # derivatives of Chebyshev polynomials.

    def get_eqn_block(self, eqn, kind='bvp'):
        """
        Build a block for a given equation and variable, sub element in left-hand side matrix
        """

        xi = self.cheby.xi
        nxblocks = len(self.degree)
        block0 = np.zeros((len(xi), self.N))
        block_empty = np.array([[]]*len(xi))

        def block(pqr):
            # This block evaluates any combination of the equation at a given degree l using a set of three coefficients that include the coupling.
            B = []
            [B.append( pqr[0]*self.cheby.d2Tn_x2(n) + pqr[1]*self.cheby.dTn_x(n) + pqr[2]*self.cheby.Tn(n) ) for n in self.n]

            # The output should have size (len(xi), N).
            return np.array(B).T

        # Concatenate horizontally to create the columns of BigL. Each row represents the coupled ODE at a given degree l.
        # The list eqn(l) has l-1 coupling on index 0, and l+1 coupling on index 2.
        PQR0 = eqn(2)
        PQRend = eqn(self.degree[-1])
        first_row = np.concatenate( [ block(PQR0[1]) , block(PQR0[2]), np.concatenate([block0] * (nxblocks-2), axis=1) ], axis=1 )
        last_row = np.concatenate( [ np.concatenate([block0] * (nxblocks-2), axis=1 ), block(PQRend[0]), block(PQRend[1]) ], axis=1 )
 
        middle_rows = []
        for row in range(0, len(self.degree)-2):
            PQR = eqn(row+3)
            middle_rows.append( np.concatenate([ (np.concatenate([block0] * row, axis=1) if row != 0 else block_empty), block(PQR[0]), block(PQR[1]), block(PQR[2]), (np.concatenate([block0] * (nxblocks-3-row), axis=1) if (nxblocks-3-row) != 0 else block_empty ) ], axis=1 ) ) 

        # Concatenate vertically to assamble the columns of BigL into a matrix
        L = np.concatenate( [first_row, np.concatenate(middle_rows, axis=0), last_row], axis=0 )
        
        return L

    def get_BigL(self, kind='bvp-core'):
        """
        ensamble the left-hand side matrix of the problem. 
        """
        block = self.get_eqn_block
        
        block(self.eqn1_y3)
        # Concatenate equation blocks for each equation and variable.
        row1 = np.concatenate([block(self.eqn1_y1), block(self.eqn1_y3), block(self.eqn1_psi)], axis=1) 
        row2 = np.concatenate([block(self.eqn2_y1), block(self.eqn2_y3), block(self.eqn_zeros)], axis=1)
        row3 = np.concatenate([block(self.eqn3_y1), block(self.eqn3_y3), block(self.eqn_zeros)], axis=1)
        BigL = np.concatenate( [row1, row2, row3 ])
        
        self.add_bc(BigL, kind=kind)

        return 
    
    def get_bc_block(self, bc='bot'):
        """
        Bottom boundary condition with a rigid core (no-slip).
        bc = {'bot', 'top'}
        """
        nbc = len(self.degree)
        def bcb(l, spec=None):
            b = []
            if bc == 'bot':
                [b.append( self.cheby.Tn_bot(n) ) for n in self.n]

            elif bc == 'top':
                [b.append( self.cheby.Tn_top(n) ) for n in self.n]
            
            elif bc == 'top_dp':
                [b.append( self.cheby.Tn_top(n)*(self.g - 4*np.pi*self.G*self.rho*self.Rp/(2*l+1)) ) for n in self.n]
            
            elif bc == 'bot_dx':
                [b.append( self.cheby.dTndx_bot(n) ) for n in self.n]

            elif bc == 'top_dx':
                [b.append( self.cheby.dTndx_top(n) ) for n in self.n]

            elif bc == 'top_sf':
                """
                Stress free bc at the top
                """
                [b.append( self.cheby.dTndx_top(n)/self.x2 - self.cheby.Tn_top(n)/self.x2**2 ) for n in self.n]
            
            elif bc == 'bot_sf':
                """
                Stress free bc at the bot
                """
                [b.append( self.cheby.dTndx_bot(n)/self.x1 - self.cheby.Tn_bot(n)/self.x1**2 ) for n in self.n]

            elif bc == 'zero':
                [b.append( 0 ) for n in self.n]

            elif spec == 'y3_minus':
                [b.append( (l-1)/l*self.Qlm(l)*self.cheby.dTndx_bot(n) ) for n in self.n]

            elif spec == 'y3_plus':
                [b.append( (l+2)/(l+1)*self.Qlm(l+1)*self.cheby.dTndx_bot(n) ) for n in self.n]

            elif bc == 'y_1':
                [b.append( (1j*self.om2/2/self.OM - 1j*self.order/l/(l+1))*self.cheby.d2Tndx2_bot(n)*self.x1/l/(l+1) ) for n in self.n]


            # output is a numpy array of lenght N
            return np.array(b)
        
        if bc == 'special':
            
            first_col = np.concatenate([ np.zeros(self.N), bcb(2, spec='y3_plus'), np.concatenate([np.zeros(self.N)]*(nbc-2))   ]) 
            last_col = np.concatenate([ np.concatenate([np.zeros(self.N)]*(nbc-2)), bcb(nbc+2, spec='y3_minus') ,np.zeros(self.N)   ]) 
 
            cols = []
            [cols.append( np.concatenate([ (np.concatenate([np.zeros(self.N)]*(col-1)) if col-1 != 0 else np.array([])), bcb(col+2, 'y3_minus'), np.zeros(self.N), bcb(col+2, 'y3_plus') , (np.concatenate([np.zeros(self.N)]*(nbc-2-col)) if (nbc-2-col)  != 0 else np.array([]) )  ]) ) for col in range(1, nbc-1) ]
            
            cols.append(last_col)
            cols.insert(0, first_col)
            
            return np.array(cols)

        cols = []
        [cols.append( np.concatenate([ (np.concatenate([np.zeros(self.N)]*col) if col != 0 else np.array([])), bcb(col+2), (np.concatenate([np.zeros(self.N)]*(nbc-1-col)) if (nbc-1-col)  != 0 else np.array([]) )  ]) ) for col in range(0, nbc) ]

        return np.array(cols)

    def add_bc(self, L, kind='bvp-core'):
        """
        Add the boundary conditions to the big matrix.
        """
        
        # Vertically stack the rows containing the boundary conditions at the bottom of BigL
        if kind == 'bvp':
            # Flow covers the entire body, from center to surface
            print('Solving core with alternative bottom bc')
            self.BigL = np.vstack( (L, self.centerBc(), self.BigflowBc(kind='top')) )  

        elif kind == 'bvp-core':
            # Flow is restricted to a shell (i.e., global ocean)
            
            row1 = np.concatenate( [self.get_bc_block(bc='bot'), self.get_bc_block(bc='zero'), self.get_bc_block(bc='zero')], axis=1) # y1 = 0 at ocean bottom
            row2 = np.concatenate( [self.get_bc_block(bc='zero'), self.get_bc_block(bc='bot_sf'), self.get_bc_block(bc='zero')], axis=1) # y3 = 0 at ocean bottom
            row4 = np.concatenate( [self.get_bc_block(bc='top_dp'), self.get_bc_block(bc='zero'), self.get_bc_block(bc='top')], axis=1) # y1 at surface
            
            # from continuity 
            #row5 = np.concatenate( [self.get_bc_block(bc='bot_dx'), self.get_bc_block(bc='zero'), self.get_bc_block(bc='zero')], axis=1)
            row5 = np.concatenate( [self.get_bc_block(bc='zero'), self.get_bc_block(bc='zero'), self.get_bc_block(bc='bot')], axis=1)
            #row5 = np.concatenate( [self.get_bc_block(bc='zero'), self.get_bc_block(bc='top'), self.get_bc_block(bc='zero')], axis=1)

            row3 = np.concatenate( [self.get_bc_block(bc='zero'), self.get_bc_block(bc='top_sf'), self.get_bc_block(bc='zero')], axis=1) # 
            #row3 = np.concatenate( [self.get_bc_block(bc='y_1'), self.get_bc_block(bc='special'), self.get_bc_block(bc='zero')], axis=1) # 
            # from eq 1.
            row6 = np.concatenate( [self.get_bc_block(bc='zero'), self.get_bc_block(bc='zero'), self.get_bc_block(bc='bot_dx')], axis=1) # dpsi at ocean bottom


            self.BigL = np.concatenate( [L, row1, row2, row3, row5, row6, row4], axis=0 )  

        return
    
    def get_Bigf(self, kind='bvp'):
        """
        Build the problem right-hand side matrix
        """

        m = self.order

        norm = self.Rp/np.pi
        
        # NOTE: both the evanescent central region (core) and central regularity have fbc=0.
        FBCtop = [] 
        for l in self.degree: 
            # add the forcing (right-hand side) in the outer boundary condition (dp=0)
            # Added a fix to dissipation.

            FBCtop.append( self.Ulm(l,m) )
        
        # Concatenate the forcing and right-hand side of inner and outer boundary conditions.
        self.Bigf = np.concatenate( (self.f(), np.zeros(5*len(self.degree)), FBCtop ) )

    # End building matrices
    ## ----------------------------------------------------------------------------
    
    def solve(self, kind='bvp'):
        """
        Solve the problem
        """

        self.get_Bigf(kind)

        self.get_BigL(kind)
        
        self.an = self.cheby.lin_solve( self.BigL, self.Bigf, sparse=True)

        ans = np.split(self.an, 3)

        self.y1, self.dy1, self.d2y1 = self.u(ans[0])
        self.y3 = self.u(ans[1])[0]
        self.psi, self.dpsi, self.d2psi = self.u(ans[2])

        self.get_y2()
        
        self.get_love()
        
        self.get_power()

        return 

    def load(self, an, kind='bvp'):
        """
        Load a saved coefficients structure

        It needs update
        """

        self.get_Bigf(kind)

        self.get_BigL(kind)
        
        self.an = an 

        self.psi, self.dpsi, self.d2psi = self.u()

        self.get_y2()

        self.get_love()

        return 

    # Evaluate love numbers
    def get_love(self):
        """
        Solve for the love number. We have two contributions: the boundary displacement and internal displacement. 

        """
        m = self.order

        for i in range(0,len(self.degree)):
            l = self.degree[i]
            
            # gravity is obtained only at the surface
            indSurface = np.argmax(self.cheby.xi)
           
            # Gravity from the surface displacement
            phi = 4*np.pi*self.G*self.rho*self.Rp/(2*l+1)*self.y1[i][indSurface]
            
            # Gravity from the interior displacement.
            #rhop = self.rho*self.y1*self.N2/self.g
            # TBD. Include the solution to Poisson's equation

            k = phi/self.Ulm(l,m) 

            dk = (k-self.k_hydro(l))/self.k_hydro(l)

            self.dk.append(dk)
            self.k.append(k)
            self.phi.append( phi )

            # It only works well when out of resonance.
            self.Q.append( abs(np.absolute(k)/np.imag(k)) )

            """
            sanity check

            5/2*self.Rp*self.om*(2*self.OM+self.om)/2/self.g*2.5*self.Ulm(l,m)
            dk = -5/2*self.psi[i][indSurface]/2.5/self.Ulm(l,m)*100
            """

        return

    def get_power(self):
        """
        Calculate the power dissipated by integrating the flow
        """
    
        for i in range(0,len(self.degree)):
            l = self.degree[i]
            
            gamma = 1/self.tau
            r = self.Rp*self.cheby.xi/np.pi

            En = gamma*self.rho*self.om**2*np.trapz(r**2*abs(self.y1[i])**2 + r**2*l*(l+1)*(abs(self.y2[i])**2 + abs(self.y3[i])**2), x=r)  

            self.E.append( abs(En) )

        return

    def get_y2(self):
        """
         Build the missing spheroidal component of displacement using continuity 

         NOTE: The derivatives obtained from the solver correspond to derivatives w.r.t. variable 'xi'
        """

        r = self.Rp*self.cheby.xi/np.pi
        y2 = []

        for i in range(0,len(self.degree)):
            l = self.degree[i]
            y2.append( (r**2*self.dy1[i]*np.pi/self.Rp+ 2*r*self.y1[i])/l/(l+1)/r )

        self.y2 = y2

        return

    # Build the Chebyshev solution from an array of Chebyshev coefficients. 
    def u(self, an):
        pol = []
        [pol.append(self.cheby.Tn(n)) for n in self.n]
        u = []
        [ u.append( np.dot(np.array(pol).T, a)) for a in np.split(an, len(self.degree)) ]
        
        pol = []
        [pol.append(self.cheby.dTn_x(n)) for n in self.n]
        du = []
        [ du.append( np.dot(np.array(pol).T, a)) for a in np.split(an, len(self.degree)) ]
        
        pol = []
        [pol.append(self.cheby.d2Tn_x2(n)) for n in self.n]
        d2u = []
        [ d2u.append( np.dot(np.array(pol).T, a)) for a in np.split(an, len(self.degree)) ]

        return [u, du, d2u]
