import pint
import numpy as np
import scipy
from scipy.optimize import root
from scipy.integrate import quad
import matplotlib.pyplot as plt
from iapws import IAPWS97 as ST
import pandas as pd


class SMR(object):
    """ Class used for solving boiling SMR with natural circulation

        --- integrations of longer length done with trapezoidal integration instead
            of scipy.optimize.quad to improve runtimes (accuracy of integration
            altered by mesh_size attribute)

        --- All necessary values have units and units are carried through each step

        --- Inputs of any units can be used
    """
    def __init__(self, power=None, height=None, pitch=None, T_inf=None,
                PF_power=None, PF_axial=None, D_clad=None, c_thick=None,
                D_pellet=None, k_c=None, n_rods=None, hg=None, pressure=None,
                G=None, gamma=None, cp=None, U=None, T_sat=None, channel='average',
                heat_flux=None, h_in=None, method='thom',life='MOC', CPR=None,
                core_height=None, D_clad_inner=None, W_can=None,
                n_w_rods=None, n_grids=None, k_grid=None, T_fuel_max=None,
                k_core_in=None, k_core_out=None, k_dc=None, k_dry_in=None,
                k_dry_out=None, D_vessel=None, D_chimney=None, n_dryers=None,
                A_sep=None, L_sep=None, L_dc=None):

        # Defining all initial class attributes (variables) to be used throughout code
        self.U = U
        self.core_height, self.D_clad_inner, self.W_can = core_height, D_clad_inner, W_can
        self.n_w_rods, self.n_grids, self.k_grid, self.T_fuel_max = n_w_rods, n_grids, k_grid, T_fuel_max
        self.k_core_in, self.k_core_out, self.k_dc, self.k_dry_in = k_core_in, k_core_out, k_dc, k_dry_in
        self.k_dry_out, self.D_vessel, self.D_chimney, self.n_dryers = k_dry_out, D_vessel, D_chimney, n_dryers
        self.A_sep, self.L_sep, self.L_dc = A_sep, L_sep, L_dc
        self.gamma, self.life,self.CPR  = gamma, life, CPR
        self.height = height.to(self.U.inch)
        self.pitch = pitch.to(self.U.inch)
        self.PF_axial = PF_axial
        self.D_clad = D_clad.to(self.U.inch)
        self.power = power.to(self.U.MW)
        self.PF_power = PF_power
        self.n_rods = n_rods
        self.k_c = k_c.to(self.U.Btu/(self.U.hour*self.U.feet*self.U.rankine))
        self.hg = hg.to(self.U.Btu/(self.U.hour*self.U.feet**2*self.U.rankine))

        try:
            if D_pellet == None:
                D_pellet = D_clad-2*c_thick
                self.D_pellet = D_pellet.to(self.U.inch)
            else:
                self.D_pellet = D_pellet.to(self.U.inch)
        except:
            pass

        # Finding necessary areas and equivalent diameters
        self.Ax = (pitch**2 - np.pi*(D_clad**2)/4)
        self.A_assembly = W_can**2 - 92*np.pi*(D_clad**2)/4
        self.A_core = self.A_assembly * 45
        self.A_dc = np.pi/4*(self.D_vessel**2-self.D_chimney**2)
        self.De_dc = self.D_vessel-self.D_chimney
        self.De_sep = ((self.A_sep*4/np.pi)**0.5)
        self.De = (4*(pitch**2 - np.pi*(D_clad**2)/4)) / (np.pi * D_clad)
        self.A_ch = np.pi * self.D_chimney**2/4

        # Defining useful terms for later calculations
        self.pressure = pressure.to(self.U.psi)
        self.p_crit = 3206 * self.U.psi
        self.h0_guess = 5
        self.g = 32.2 * U.foot / U.s**2
        self.mesh_size = 10
        self.z_mesh = np.linspace(0, self.height.magnitude, self.mesh_size)

        # Determining feed inlet enthalpy
        obj = ST(P=pressure.to(self.U.MPa).magnitude, T=T_inf.to(self.U.degK).magnitude)
        self.h_feed_in = (obj.h * self.U.kJ / self.U.kg).to(self.U.Btu/self.U.lb)
        self.rho_in = (1/(obj.v * (self.U.m**3/self.U.kg))).to(self.U.lb/self.U.foot**3)

        # Finding all state values for saturation etc.
        obj = ST(P=pressure.to(self.U.MPa).magnitude, x=0)
        self.T_sat = (obj.T * self.U.degK).to(self.U.degR)
        self.rho_f = (1/(obj.v * (self.U.m**3/self.U.kg))).to(self.U.lb/self.U.foot**3)
        self.sigma = (obj.sigma * self.U.N / self.U.m).to(self.U.lbf/self.U.foot)
        self.h_f = (obj.h * self.U.kJ / self.U.kg).to(self.U.Btu/self.U.lb)
        self.mu_f = (obj.mu * self.U.Pa * self.U.s).to(self.U.lb/(self.U.foot*self.U.hour))
        self.k_f = (obj.k * self.U.W / (self.U.m * self.U.degK)).to(self.U.Btu/self.U.hour/self.U.foot/self.U.degR)
        self.cp_f = (obj.cp*U.kJ/U.degK/U.kg).to(self.U.Btu/(self.U.lb*self.U.degR))
        self.prandt = obj.Prandt
        self.cp = (obj.cp*U.kJ/U.degK/U.kg).to(self.U.Btu/(self.U.lb*self.U.degR))
        self.mu = obj.mu * self.U.Pa * self.U.s
        self.k_water = obj.k * self.U.W / (self.U.m * self.U.degK)

        obj = ST(P=pressure.to(self.U.MPa).magnitude, x=1)
        self.rho_g = (1/(obj.v * (self.U.m**3/self.U.kg))).to(self.U.lb/self.U.foot**3)
        self.h_g = (obj.h * self.U.kJ / self.U.kg).to(self.U.Btu/self.U.lb)
        self.mu_g = (obj.mu * self.U.Pa * self.U.s).to(self.U.lb/(self.U.foot*self.U.hour))
        self.h_fg = self.h_g - self.h_f
        self.v_fg = 1/(self.rho_f-self.rho_g)
        self.gc = 32.2 * (self.U.lb*self.U.foot)/(self.U.lbf*self.U.s**2)
        self.method = method
        self.shape = 1

        # Finds the lambda for beginning of cycle
        def trial(x):
            return x*np.cos(x) + np.sin(x)
        x1 = root(trial, 2).x[0]
        self.shape = np.sin(x1) * x1
        def q_z2(z, lambda_):
            return (np.pi*(self.height.magnitude+lambda_-z) / (self.height.magnitude + 2*lambda_))*np.sin(np.pi*(self.height.magnitude+lambda_-z) / (self.height.magnitude + 2*lambda_))
        def find_lambda(lambda_):
            return self.PF_axial - ((self.shape*self.height.magnitude)/(quad(q_z2, 0, self.height.magnitude, args=(lambda_))[0]))
        self.lambda_ = root(find_lambda, 1).x[0] * self.U.inch

        # defining roughness and calculating mass flow rate of feed from inputs
        self.epsilon1 = .000005*self.U.foot
        self.epsilon2 = .00015*self.U.foot
        self.m_dot_fd = self.power / (self.h_g - self.h_feed_in)

    #######################################################################################################################################################
    # Begin Fuctional Programming
    #######################################################################################################################################################

    def global_avg_params(self, G):
        """ Call function to change all global variables to average channel state """
        obj.mesh_size = 20
        self.q_dprime = ((self.gamma * self.power) / (self.shape * self.n_rods * np.pi * self.D_clad * self.height)).to(self.U.Btu/self.U.foot**2/self.U.hour)
        self.G_core = G * self.U.lb/self.U.hour/self.U.foot**2
        self.h_0 = self.find_h0().to(self.U.inch)
        self.h2_phase = self.h_2phase()
        self.m_dot = (self.G_core * self.A_core).to(self.U.lb/self.U.hour)
        self.beta = self.find_beta
        self.G_ch = self.m_dot / (self.D_chimney**2/4*np.pi)
        return print('All global variables changed to average')


    def global_hot_params(self, G):
        """ Call function to change all global variables to hot channel state """
        obj.mesh_size = 20
        self.q_dprime = ((self.gamma * self.power * self.PF_power) / (self.shape * self.n_rods * np.pi * self.D_clad * self.height)).to(self.U.Btu/self.U.foot**2/self.U.hour)
        self.G_core = G * self.U.lb/self.U.hour/self.U.foot**2
        self.h_0 = self.find_h0().to(self.U.inch)
        self.h2_phase = self.h_2phase()
        self.m_dot = (self.G_core * self.A_core).to(self.U.lb/self.U.hour)
        self.beta = self.find_beta
        self.G_ch = self.m_dot / (self.D_chimney**2/4*np.pi)
        return print('All global variables changed to hot')


    def find_G_avg(self, G):
        """ Function that returns 0 if CPR value is exactly met from input average mass flux
            Can be used with scipy.optimize.root finding function (longer runtimes) or guess value"""
        self.q_dprime = ((self.gamma * self.power) / (self.shape * self.n_rods * np.pi * self.D_clad * self.height)).to(self.U.Btu/self.U.foot**2/self.U.hour)
        self.G_core = G * self.U.lb/self.U.hour/self.U.foot**2
        G_avg = self.G_core
        self.h_in = self.find_h_in(G_avg)
        self.h_0 = self.find_h0().to(self.U.inch)
        P_original = self.P_core()
        G_hot = root(self.find_G_hot, 1*10**6, args=(P_original)).x[0] * self.U.lb/self.U.hour/self.U.foot**2
        return self.CPR - (self.q_crit() / self.q_dprime).to(self.U.dimensionless).magnitude


    def find_G_hot(self, G, P_original):
        """ Finds G_hot that gives equivalent pressure drop across the core as G_avg
           (Do not call function directly, is referenced in (find_G_avg function) """
        self.q_dprime = ((self.gamma * self.power * self.PF_power) / (self.shape * self.n_rods * np.pi * self.D_clad * self.height)).to(self.U.Btu/self.U.foot**2/U.hour)
        self.G_core = G * self.U.lb/self.U.hour/self.U.foot**2
        try:
            self.G_core = self.G_core[0]
        except:
            pass
        self.h_0 = self.find_h0().to(self.U.inch)
        return (P_original.to(self.U.psi) - self.P_core().to(self.U.psi)).magnitude


    def find_inlet_conditions(self):
        """ Finds inlet temperature to core from inlet enthalpy (h_in must be found first) """
        obj = ST(P=self.pressure.to(self.U.MPa).magnitude, h=self.h_in.to(self.U.kJ/self.U.kg).magnitude)
        self.T_inf=(obj.T*self.U.degK).to(self.U.degR)
        self.rho_in = (1/(obj.v * (self.U.m**3/self.U.kg))).to(self.U.lb/self.U.foot**3)


    def find_h_in(self, G_avg):
        """ Finds the inlet enthalpy based upon G_avg """
        return self.h_feed_in * (self.m_dot_fd/(G_avg*self.A_core)) + self.h_f * (1 - self.m_dot_fd/(G_avg*self.A_core))


    def q_z(self, z):
        """ Returns the heat flux shape function """
        return ((np.pi*(self.height+self.lambda_-z*self.U.inch) / (self.height + 2*self.lambda_))
                   * np.sin(np.pi*(self.height+self.lambda_-z*self.U.inch) / (self.height + 2*self.lambda_)))


    def h_z(self, z):
        """" Return enthalpy at an axial location in core """
        z_mesh = np.linspace(0, z, self.mesh_size)
        int = np.trapz([self.q_z(i) for i in z_mesh], z_mesh)
        return (self.h_in + (1/(self.G_core*self.Ax*self.gamma)*np.pi*self.D_clad*self.q_dprime*
            int*self.U.inch))


    def conduct_int(self, T):
        """ Conductivity Integral,
           -not to be called directly """
        return 3978.1/(692.6+T) + 6.02366*10**(-12) * (T+460)**3


    def fuel_root(self, T, T_sfuel, z):
        """ Returns conductivty integral eqn. to be solved iteratively,
            -not to be called directly """
        T_mesh = np.linspace(T_sfuel, T, self.mesh_size)
        int = np.trapz([self.conduct_int(i) for i in T_mesh], T_mesh)
        return ((int*self.U.Btu/(self.U.foot*self.U.hour) -
            ((self.q_dprime*self.q_z(z)*2*self.D_clad/2/(self.D_pellet/2)**2)*(self.D_pellet/2)**2/4)))


    def T_coolant(self, z):
        """ Returns the coolant temperature at axial location """
        T = (((1/(self.G_core*self.Ax*self.cp)*np.pi*self.D_clad*self.q_dprime*quad(self.q_z, 0, z,
                )[0]*self.U.inch).to(self.U.degR) + self.T_inf) )
        if T > self.T_sat:
            return self.T_sat
        else:
            return T


    def T_clad(self, z):
        """ Returns the clad temperature - can also call T_clad_chen directly """
        return self.T_clad_chen(z)


    def T_sfuel(self, z):
        """ Returns the fuel surface temperature """
        return ((self.T_clad(z).magnitude
                + (self.q_dprime*self.q_z(z) * self.D_clad/2 *
                (1/(self.hg*self.D_pellet/2) + 1/self.k_c*np.log(self.D_clad/self.D_pellet))).to(self.U.degR).magnitude) * self.U.degR)


    def T_fuel(self, z):
        """ Returns the fuel centerline temperature """
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
        return (((root(self.fuel_root, 2000, args=(self.T_sfuel(z).to(self.U.degF).magnitude, z)).x[0])+458.67) * self.U.degR)


    def max_T_fuel(self):
        """ Finds max fuel centerline temperature and location """
        zrange = np.linspace(0, self.height.magnitude, self.height.magnitude+1)
        temp = []
        for i in zrange:
            temp.append(self.T_fuel(i))
        return max(temp), temp.index(max(temp)) *self.U.inch


    def quality(self, z):
        """ Returns the quality at a point """
        q = ((self.h_z(z)-self.h_f)/(self.h_fg)).to(self.U.dimensionless).magnitude
        return max(q, 0)


    def quality_2(self, z):
        """ Returns the quality at a given height """
        return ((self.h_z(z)-self.h_f)/(self.h_fg)).to(self.U.dimensionless).magnitude


    def martinelli(self, z):
        """ Returns martinelli for chen """
        return ((self.quality(z)/(1-self.quality(z)))**.9*(self.rho_f/self.rho_g)**.5*(self.mu_g/self.mu_f)**.1).magnitude


    def S(self, z):
        """ Returns S for chen """
        return (.9622-.5822*np.arctan(self.reynolds_lo(z)*self.F(z)**1.25/(6.18*10**4))).magnitude


    def reynolds_lo(self, z):
        """ Returns the reynolds lo value  """
        return self.G_core*(1-self.quality(z))*self.De/self.mu


    def F(self, z):
        """ Returns factor F for chen """
        if self.martinelli(z) <= .1:
            return 1
        else:
            return 2.35*(self.martinelli(z)+.213)**.736


    def h_lo_dittus(self, z):
        """" Returns h_lo weismann for chen """
        h = .023*(self.G_core*(1-self.quality(z))*self.De/self.mu)**.8*(self.prandt)**0.4*(self.k_water/self.De)*self.F(z)
        return h.to(self.U.Btu/self.U.hour/self.U.foot**2/self.U.degR)


    def h_lo_weismann(self, z):
        """ Returns h_lo weismann for chen """
        self.reynolds_core = ((self.G_core*self.De/self.mu).to(self.U.dimensionless)).magnitude
        h = (.042*self.pitch/self.D_clad-.024)*self.reynolds_core**.8*self.prandt**(1/3)*(self.k_water/(4*(self.pitch**2-np.pi*(self.D_clad**2/4))/(np.pi*self.D_clad)))*self.F(z)
        return h.to(self.U.Btu/self.U.hour/self.U.foot**2/self.U.degR)


    def h_2phase(self):
        """ Returns the nucleate boiling h_2phase """
        J=778*self.U.foot*self.U.lb/self.U.Btu
        top = self.k_f**.79*self.cp_f**.45*self.rho_f**.49 * self.gc**0.25
        bottom = self.sigma**0.5*self.mu_f**.29*self.h_fg**.24*self.rho_g**.24
        term_1 = ((self.h_fg * J)/(self.T_sat*self.v_fg))**.75
        return (.00122*(top/bottom)*term_1)


    def root_chen(self, Tw, z):
        """ Returns root for chen """
        if self.method == 'chen-dittus':
            tmp = (self.S(z)*self.h2_phase*(Tw*self.U.degR-self.T_sat)**1.99).magnitude * (self.U.Btu/self.U.hour/self.U.foot**2)
            return self.h_lo_dittus(z)*(Tw*self.U.degR-self.T_coolant(z)) + tmp - self.q_dprime*self.q_z(z)
        elif self.method == 'chen-weismann':
            tmp = (self.S(z)*self.h2_phase*(Tw*self.U.degR-self.T_sat)**1.99).magnitude * (self.U.Btu/self.U.hour/self.U.foot**2)
            return self.h_lo_weismann(z)*(Tw*self.U.degR-self.T_coolant(z)) + tmp - self.q_dprime*self.q_z(z)


    def T_clad_chen(self, z):
        """ Returns the T_clad for chen """
        return root(self.root_chen, 1100, args=(z)).x[0]*self.U.degR


    def Dix(self, z):
        """ Returns the Dix correlation alpha for standard quality equation (different
            than Dix_sep function below) """
        try:
            b = (self.rho_g / self.rho_f)**(0.1)
            beta = self.quality(z) / (self.quality(z) + (1-self.quality(z)) * (self.rho_g/self.rho_f))
            V_gj = 2.9 * (((self.sigma*self.g*(self.rho_f-self.rho_g)) / (self.rho_f**2))**(1/4))
            Co = beta * (1+((1/beta)-1)**b)
            return ( (1) / (Co*(1 + ((1-self.quality(z))*self.rho_g)/(self.quality(z)*self.rho_f)) + (self.rho_g*V_gj) / (self.G_core*self.quality(z))) )
        except:
            return 0


    def Dix_sep(self, z):
        """ Returns the Dix correlation alpha strictly for separator quality """
        try:
            b = (self.rho_g / self.rho_f)**(0.1)
            beta = self.quality_sep(z) / (self.quality_sep(z) + (1-self.quality_sep(z)) * (self.rho_g/self.rho_f))
            V_gj = 2.9 * (((self.sigma*self.g*(self.rho_f-self.rho_g)) / (self.rho_f**2))**(1/4))
            Co = beta * (1+((1/beta)-1)**b)
            return ( (1) / (Co*(1 + ((1-self.quality_sep(z))*self.rho_g)/(self.quality_sep(z)*self.rho_f)) + (self.rho_g*V_gj) / (self.G_core*self.quality_sep(z))) )
        except:
            return 0


    def density(self, x):
        """ Returns the mixture density in the core """
        if x <= self.h_0.magnitude:
            en = ((self.h_z(x).to(self.U.kJ/self.U.kg) - self.quality(x)*self.h_g)/(1-self.quality(x))).magnitude
            stuff = ST(h=en, P=self.pressure.to(self.U.MPa).magnitude)
            return (1/(stuff.v * (self.U.m**3/self.U.kg))).to(self.U.lb/self.U.foot**3).magnitude
        else:
            return (self.Dix(x) * self.rho_g + (1-self.Dix(x)) * self.rho_f).magnitude


    def psi(self, z):
        """ 2-phase multiplication term for local loss pressures """
        if z > self.h_0.magnitude:
            return 1 + (self.rho_f/(self.rho_f-self.rho_g))*self.quality(z)
        else:
            return 1


    def p2m(self, z):
        """ 2-phase multiplication term for friction loss pressures """
        if z > self.h_0.magnitude:
            return (1 + 20/self.martinelli_2(self.quality(z)) + 1/self.martinelli_2(self.quality(z))**2)*(1-self.quality(z))**1.8
        else:
            return 1

    def martinelli_2(self, x):
        """ Martinelli parameter for 2-phase multiplication term
            (Not the same as for chen correlation)"""
        if x > 0:
            return ((self.mu_f/self.mu_g)**0.2 * ((1-x)/x)**1.8 * (self.rho_g/self.rho_f))**0.5
        else:
            return 1


    def root_friction(self, f, epsilon, D, Re):
        """ Returns the friction equation to be solved iteratively """
        return 1.14-2*np.log10(epsilon/D+(9.35)/(Re*np.sqrt(f)))-1/np.sqrt(f)


    @property
    def a_crit(self):
        """ Returns a for critical problem """
        p_crit = ((self.p_crit).to(self.U.MPa)).magnitude
        p = ((self.pressure).to(self.U.MPa)).magnitude
        G = ((self.G_core).to(self.U.kg/self.U.m**2/self.U.s)).magnitude
        tmp = (1-p/p_crit)**3
        if G <= 3375 * tmp:
            return (tmp/(1+1.481*10**(-4)*G))*(self.D_clad/self.De)
        else:
            return (tmp**(1/3)/(G/1000)**(1/3))*(self.D_clad/self.De)


    @property
    def b_crit(self):
        """ Return b for citical problem """
        p_crit = ((self.p_crit).to(self.U.MPa)).magnitude
        p = ((self.pressure).to(self.U.MPa)).magnitude
        G = ((self.G_core).to(self.U.kg/self.U.m**2/self.U.s)).magnitude
        D = ((self.D_clad).to(self.U.m)).magnitude
        return (.199*(p_crit/p - 1)**(0.4)*G*D**1.4)*self.U.m


    def root_h0(self, h0):
        """ Returns root for h0 value for CISE correlation """
        left = (self.a_crit*(self.height-h0*self.U.inch))/(self.b_crit+(self.height-h0*self.U.inch))
        right1 = (self.h_f-self.h_in)/self.h_fg
        right2 = quad(self.q_z, h0, self.height.magnitude)[0]
        right3 = quad(self.q_z, 0, h0)[0]
        return left - right1*(right2/right3)


    def find_h0(self):
        """ Returns the h0 value for CISE correlation """
        h = root(self.root_h0, self.h0_guess).x[0] * self.U.inch
        if h.magnitude < 0:
            return 0 * self.U.inch
        else:
            return h


    def q_crit(self):
        """ Returns the q_crit value for CISE correlation """
        top = self.G_core*self.Ax*self.gamma*(self.h_f-self.h_in)
        bottom = np.pi*self.D_clad*quad(self.q_z, 0, self.h_0.magnitude)[0]*self.U.inch
        return top/bottom


    def P_core(self):
        """ Finds the total pressure drop across the core """
        alpha_g = self.Dix(self.height.magnitude)
        alpha_l = 1-alpha_g
        term1 = self.G_core**2*((1-self.quality(self.height.magnitude))**2/(alpha_l*self.rho_f) + self.quality(self.height.magnitude)**2/(alpha_g*self.rho_g)-1/self.rho_in)
        self.Re_core = (self.G_core * self.De/self.mu).to(self.U.dimensionless).magnitude
        self.fr_core = root(self.root_friction, .01, args=(self.epsilon1,self.De,self.Re_core)).x[0]
        mesh1 = np.linspace(self.h_0.magnitude, self.height.magnitude, self.mesh_size)
        int1 = np.trapz([self.p2m(i) for i in mesh1], mesh1)
        term2 = (self.fr_core*self.h_0/self.De * self.G_core**2/(2*self.rho_f) + self.fr_core*self.G_core**2/(self.De*2*self.rho_f) * int1*self.U.inch)
        grid = np.linspace(0, self.height.magnitude, self.n_grids+2)
        term3 = ((self.G_core**2/2 * (sum([self.k_grid*self.psi(i)/self.density(i) for i in grid[1:-1]]) + self.k_core_in**self.psi(0)/self.density(0) +
                self.k_core_out**self.psi(self.height.magnitude)/self.density(self.height.magnitude))/ (self.U.lb/self.U.foot**3)))
        int2 = np.trapz([self.density(i) for i in self.z_mesh], self.z_mesh)
        term4 = self.g * int2 * self.U.lb/self.U.foot**3 * self.U.inch
        return (term1 + term2 + term3 + term4).to(self.U.psi)


    def P_ch(self, H):
        """ Finds the total pressure drop across the chimney """
        H = H * self.U.inch
        self.Re_ch = (self.G_ch * self.D_chimney/self.mu).to(self.U.dimensionless).magnitude
        self.fr_ch = root(self.root_friction, .01, args=(self.epsilon1,self.De,self.Re_core)).x[0]
        term1 = (self.fr_ch*H*self.G_ch**2) / (self.D_chimney*2*self.rho_f)*self.p2m(H.magnitude)
        term2 = self.density(self.height.magnitude) * self.U.lb/self.U.foot**3 * self.g * self.height
        return (term1 + term2).to(self.U.psi)


    def P_dc(self, H):
        """ Finds the total pressure drop across the downcomer """
        H = H * self.U.inch
        self.G_dc = self.m_dot / self.A_dc
        self.Re_dc = (self.G_dc * self.De_dc / self.mu).to(self.U.dimensionless).magnitude
        self.fr_dc = root(self.root_friction, .01, args=(self.epsilon2,self.De_dc,self.Re_dc)).x[0]
        term1 = self.fr_dc*(H + self.height)/self.De_dc *  self.G_dc**2/(2*self.rho_in)
        term2 = self.k_dc * self.G_dc**2/(2*self.rho_in) - self.rho_in * self.g * (H + self.height + self.L_dc)
        return (term1 + term2).to(self.U.psi)


    def P_sep(self):
        """ Finds the total pressure drop across the separators """
        term1 = self.G_sep(self.L_sep.magnitude)**2/self.rho_sep(self.L_sep.magnitude)
        term2 = self.G_sep(0)**2/self.rho_sep(0)
        x_range1 = np.linspace(0, self.L_sep.magnitude, self.mesh_size)
        term3 = np.trapz([self.sep_int1(i) for i in x_range1], x_range1) * self.U.psi
        term4 = (self.G_sep(0)**2/2*self.k_dry_in*self.psi(0)/self.rho_f +
            self.G_sep(self.L_sep.magnitude)**2/2*self.k_dry_out*self.psi(self.L_sep.magnitude)/self.rho_f)
        term5 = np.trapz([self.sep_int2(i) for i in x_range1], x_range1) * self.U.psi
        return (term1 - term2 + term3 + term4 + term5).to(self.U.psi)


    @property
    def find_beta(self):
        """ Finds the constant beta term for separator """
        return -np.log( (1-.9975) / (1-self.quality(self.height.magnitude)) )


    def quality_sep(self, z):
        """ Returns the quality for the separator """
        return 1 - ((1-.9975) / np.exp(-self.beta))


    def rho_sep(self, z):
        """ Returns the density at a point in the separator """
        return (1 / (1/self.rho_f*(1-self.quality_sep(z))+self.quality_sep(z)*1/self.rho_g))


    def sep_int1(self, z):
        """ Returns the function used for integration in the separator
            (strictly the friction term) """
        self.Re_sep = (self.G_sep(z) * self.De_sep / self.mu).to(self.U.dimensionless).magnitude
        self.fr_sep = root(self.root_friction, .01, args=(self.epsilon1,self.De_sep,self.Re_sep)).x[0] * 1.5
        return (self.fr_sep / self.De_sep * self.G_sep(z)**2/2/self.rho_f * self.p2m(z)*self.U.inch).to(self.U.psi).magnitude


    def sep_int2(self, z):
        """ Returns the function used for integration in the separator
            (strictly the elevation term) """
        return (self.rho_sep(z) * self.g * self.U.inch).to(self.U.psi).magnitude


    def G_sep(self, z):
        """ Returns the mass flux of the separator at a point """
        term1 =  (1 - self.Dix(self.height.magnitude)) * self.G_ch * self.A_ch / self.A_sep * np.exp((-self.beta*z*self.U.inch)/self.L_sep)
        term2 = self.Dix(self.height.magnitude) * self.G_ch * self.A_ch / self.A_sep
        return term1 + term2


    def P_balance(self, H):
        """ Returns the total pressure drop around the system
            = 0 when balanced """
        return self.P_core_stat + self.P_dc(H) + self.P_ch(H) + self.P_sep_stat


    def H_chimney(self):
        """ Returns the minimum height of the chimney based upon the pressure balance equation """
        obj.P_core_stat = obj.P_core()
        obj.P_sep_stat = obj.P_sep()
        return root(self.P_balance, 21*12).x[0] * self.U.inch


    def plot(self, coolant=False, clad=False, fuel=False, units=None, method=None):
        """ Returns a plot of the specified regions when set to True """
        zrange = np.linspace(0,self.height.magnitude, int(self.height.magnitude/3))
        T_coolant, T_clad, T_fuel = {}, {}, {}
        if units == None:
            units = self.U.degF
        if units == self.U.degK:
            unit_string = str(units)[-6]
        else:
            unit_string = str(units)[-1]
        if method==None:
            method == ['thom']
        for meth in method:
            T_coolant[meth], T_clad[meth], T_fuel[meth] = [], [], []
            self.method = meth
            for z in zrange:
                if coolant:
                    T_coolant[meth].append(self.T_coolant(z).to(units).magnitude)
                if clad:
                    T_clad[meth].append(self.T_clad(z).to(units).magnitude)
                if fuel:
                    T_fuel[meth].append(self.T_fuel(z).to(units).magnitude)
        plt.figure(figsize=(10,6))
        for meth in method:
            if coolant:
                plt.plot(zrange, T_coolant[meth], label='Coolant-'+str(meth))
            if clad:
                plt.plot(zrange, T_clad[meth], label='Clad-'+str(meth))
            if fuel:
                plt.plot(zrange, T_fuel[meth], label='Fuel-'+str(meth))
        plt.legend()
        plt.ylabel('Temperature [' + unit_string + ']')
        plt.xlabel('Rod Height [inches]')
        plt.show()
