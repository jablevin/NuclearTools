import pint
import numpy as np
import scipy
from scipy.optimize import root
from scipy.integrate import quad
import matplotlib.pyplot as plt
from iapws import IAPWS97 as ST
from scipy.misc import derivative

class reactor_temperatures(object):
    def __init__(self, power=None, height=None, pitch=None, T_inf=None,
                PF_power=None, PF_axial=None, D_clad=None, c_thick=None,
                D_pellet=None, k_c=None, n_rods=None, hg=None, pressure=None,
                G=None, gamma=None, cp=None, U=None, T_sat=None, channel=None,
                heat_flux=None, h_in=None, method='thom',life='MOC', CPR=None):
        self.U = U
        if power!=None:
            self.power = power.to(self.U.MW)
        if T_inf != None:
            self.T_inf = T_inf.to(self.U.degR)
        if PF_power != None:
            self.PF_power = PF_power
        if n_rods != None:
            self.n_rods = n_rods
        if c_thick != None:
            self.c_thick = c_thick.to(self.U.inch)
        if k_c != None:
            self.k_c = k_c.to(self.U.Btu/(self.U.hour*self.U.feet*self.U.rankine))
        if hg != None:
            self.hg = hg.to(self.U.Btu/(self.U.hour*self.U.feet**2*self.U.rankine))
        try:
            if D_pellet == None:
                D_pellet = D_clad-2*c_thick
                self.D_pellet = D_pellet.to(self.U.inch)
            else:
                self.D_pellet = D_pellet.to(self.U.inch)
        except:
            pass

        self.height = height.to(self.U.inch)
        self.pitch = pitch.to(self.U.inch)
        self.PF_axial = PF_axial
        self.D_clad = D_clad.to(self.U.inch)
        self.Ax = pitch**2 - np.pi*(D_clad**2)/4
        self.De = (4*(pitch**2 - np.pi*(D_clad**2)/4)) / (np.pi * D_clad)
        self.pressure = pressure.to(self.U.psi)
        self.G = G.to(self.U.lb/(self.U.hour*self.U.feet**2))
        self.gamma = gamma
        self.life = life
        self.CPR = CPR
        self.p_crit = 3206 * self.U.psi
        self.h0_guess = 20
        self.gravity = 9.81 * self.U.m/self.U.s**2
        if h_in != None:
            self.h_in = h_in
        else:
            obj = ST(P=pressure.to(self.U.MPa).magnitude, T=T_inf.to(self.U.degK).magnitude)
            self.h_in = (obj.h * self.U.kJ / self.U.kg).to(self.U.Btu/self.U.lb)


        obj = ST(P=pressure.to(self.U.MPa).magnitude, x=0)
        self.T_sat = (obj.T * self.U.degK).to(self.U.degR)
        self.rho_f = (1/(obj.v * (self.U.m**3/self.U.kg))).to(self.U.lb/self.U.foot**3)
        self.sigma = (obj.sigma * self.U.N / self.U.m).to(self.U.lbf/self.U.foot)
        self.h_f = (obj.h * self.U.kJ / self.U.kg).to(self.U.Btu/self.U.lb)
        self.mu_f = (obj.mu * self.U.Pa * self.U.s).to(self.U.lb/(self.U.foot*self.U.hour))
        self.k_f = (obj.k * self.U.W / (self.U.m * self.U.degK)).to(self.U.Btu/self.U.hour/self.U.foot/self.U.degR)
        self.cp_f = (obj.cp*U.kJ/U.degK/U.kg).to(self.U.Btu/(self.U.lb*self.U.degR))

        obj = ST(P=pressure.to(self.U.MPa).magnitude, x=1)
        self.rho_g = (1/(obj.v * (self.U.m**3/self.U.kg))).to(self.U.lb/self.U.foot**3)
        self.h_g = (obj.h * self.U.kJ / self.U.kg).to(self.U.Btu/self.U.lb)
        self.mu_g = (obj.mu * self.U.Pa * self.U.s).to(self.U.lb/(self.U.foot*self.U.hour))
        self.h_fg = self.h_g - self.h_f
        self.v_fg = 1/(self.rho_f-self.rho_g)
        self.gc = 32.2 * (self.U.lb*self.U.foot)/(self.U.lbf*self.U.s**2)
        self.method = method
        self.shape = 1

        if life == 'MOC' or self.life == 'Nominal':
            """ Finds the lambda for middle of cycle """
            def q_z2(z, lambda_):
                return np.sin(np.pi * ((z + lambda_) / (self.height.magnitude + 2*lambda_)))
            def find_lambda(lambda_):
                return self.PF_axial - (self.height.magnitude/(quad(q_z2, 0, self.height.magnitude, args=(lambda_))[0]))
            self.lambda_ = root(find_lambda, 1).x[0] * self.U.inch
        elif life == 'BOC' or self.life == 'Bottom':
            """ Finds the lambda for beginning of cycle """
            def trial(x):
                return x*np.cos(x) + np.sin(x)
            x1 = root(trial, 2).x[0]
            self.shape = np.sin(x1) * x1
            def q_z2(z, lambda_):
                return (np.pi*(self.height.magnitude+lambda_-z) / (self.height.magnitude + 2*lambda_))*np.sin(np.pi*(self.height.magnitude+lambda_-z) / (self.height.magnitude + 2*lambda_))
            def find_lambda(lambda_):
                return self.PF_axial - ((self.shape*self.height.magnitude)/(quad(q_z2, 0, self.height.magnitude, args=(lambda_))[0]))
            self.lambda_ = root(find_lambda, 1).x[0] * self.U.inch
        elif self.life == 'Top':
            """ Finds the lambda for Top Peaked Core """
            def trial(x):
                return x*np.cos(x) + np.sin(x)
            x1 = root(trial, 2).x[0]
            self.shape = np.sin(x1) * x1
            def q_z2(z, lambda_):
                return (np.pi*(lambda_+z) / (self.height.magnitude + 2*lambda_))*np.sin(np.pi*(lambda_+z) / (self.height.magnitude + 2*lambda_))
            def find_lambda(lambda_):
                return self.PF_axial - ((self.shape*self.height.magnitude)/(quad(q_z2, 0, self.height.magnitude, args=(lambda_))[0]))
            self.lambda_ = root(find_lambda, 1).x[0] * self.U.inch

        try:
            if channel == 'average':
                self.q_dprime = ((gamma * power * PF_axial) / (n_rods * np.pi * self.D_clad * height)).to(self.U.lb/self.U.second**3)
            elif channel == 'hot':
                holder = ((gamma * power) / (n_rods * np.pi * self.D_clad * height)).to(self.U.lb/self.U.second**3)
                self.q_dprime = holder * PF_power
            else:
                self.q_dprime = ((gamma * power * PF_axial) / (n_rods * np.pi * self.D_clad * height)).to(self.U.lb/self.U.second**3)
        except:
            pass

        try:
            try:
                self.heat_flux = heat_flux.to(self.U.Btu/self.U.hour/self.U.foot**2)
                def find_q_dprime(x1):
                    """ Used for finding the q_0 """
                    return ((x1/self.height) * quad(self.q_z, 0, self.height.magnitude)[0]*self.U.inch) - self.heat_flux.magnitude

                self.q_dprime = root(find_q_dprime, 150000).x[0] * (self.U.Btu/self.U.hour/self.U.foot**2)
            except:
                self.q_dprime = self.q_dprime / self.shape
        except:
            pass


        if T_inf != None:
            T_holding = (self.T_inf + self.T_sat) / 2
            obj = ST(P=pressure.to(self.U.MPa).magnitude, T=T_holding.to(self.U.degK).magnitude)
        else:
            obj = ST(P=pressure.to(self.U.MPa).magnitude, h=h_in.to(self.U.kJ/self.U.kg).magnitude)
            self.T_inf=(obj.T*self.U.degK).to(self.U.degR)

        self.prandt = obj.Prandt
        self.cp = (obj.cp*U.kJ/U.degK/U.kg).to(self.U.Btu/(self.U.lb*self.U.degR))
        self.mu = obj.mu * self.U.Pa * self.U.s
        self.k_water = obj.k * self.U.W / (self.U.m * self.U.degK)
        self.h0 = obj.h * self.U.kJ/self.U.kg
        self.reynolds = (G*self.De/self.mu)**0.8
        self.hc = (.042*pitch/D_clad-.024)*self.reynolds**.8*self.prandt**(1/3)*(self.k_water/(4*(pitch**2-np.pi*(D_clad**2/4))/(np.pi*D_clad)))
        self.hc = self.hc.to(self.U.Btu/(self.U.hour*self.U.foot**2*self.U.degR))

        try:
            T_holder = self.T_coolant(self.height)
            T_holding = (self.T_inf + T_holder) / 2
            obj = ST(P=pressure.to(self.U.MPa).magnitude, T=T_holding.to(self.U.degK).magnitude)
            self.prandt = obj.Prandt
            self.cp = (obj.cp*U.kJ/U.degK/U.kg).to(self.U.Btu/(self.U.lb*self.U.degR))
            self.mu = obj.mu * self.U.Pa * self.U.s
            self.k_water = obj.k * self.U.W / (self.U.m * self.U.degK)
            self.h0 = obj.h * self.U.kJ/self.U.kg
            self.reynolds = (G*self.De/self.mu)**0.8
            self.hc = (.042*pitch/D_clad-.024)*self.reynolds**.8*self.prandt**(1/3)*(self.k_water/(4*(pitch**2-np.pi*(D_clad**2/4))/(np.pi*D_clad)))
            self.hc = self.hc.to(self.U.Btu/(self.U.hour*self.U.foot**2*self.U.degR))
        except:
            pass

        try:
            self.zn = self.find_zn()
            self.zb_guess = self.find_zn().magnitude + 36
            self.zb = self.find_zb()
        except:
            pass

        try:
            self.x_c1 = self.x_c1()
            self.x_c2 = self.x_c2()
            self.h_0 = self.find_h0().to(self.U.inch)
            holder = self.max_Q()
        except:
            pass

        try:
            self.h2_flux = self.h_2flux()
        except:
            pass



    def q_z(self, z):
        """ Returns the heat flux depending on life cycle and height in core """
        if self.life == 'MOC' or self.life == 'Nominal':
            return np.sin((np.pi*(z*self.U.inch+self.lambda_))/(self.height+2*self.lambda_))
        elif self.life == 'BOC' or self.life == 'Bottom':
            return ((np.pi*(self.height+self.lambda_-z*self.U.inch) / (self.height + 2*self.lambda_))
                   * np.sin(np.pi*(self.height+self.lambda_-z*self.U.inch) / (self.height + 2*self.lambda_)))
        elif self.life == 'Top':
            return ((np.pi*(self.lambda_+z*self.U.inch) / (self.height + 2*self.lambda_))
                   * np.sin(np.pi*(self.lambda_+z*self.U.inch) / (self.height + 2*self.lambda_)))


    def conduct_int(self, T):
        """ Conductivity Integral,
           -not to be called directly """
        return 3978.1/(692.6+T) + 6.02366*10**(-12) * (T+460)**3


    def fuel_root(self, T, T_sfuel, z):
        """ Returns conductivty integral eqn. to be solved iteratively,
            -not to be called directly """
        return (quad(self.conduct_int, T_sfuel, T)[0]*self.U.Btu/(self.U.foot*self.U.hour) -
                ((self.q_dprime*self.q_z(z)*2*self.D_clad/2/(self.D_pellet/2)**2)*(self.D_pellet/2)**2/4))


    def T_coolant(self, z):
        """ Returns the coolant temperature """
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
        T = (((1/(self.G*self.Ax*self.cp)*np.pi*self.D_clad*self.q_dprime*quad(self.q_z, 0, z,
                )[0]*self.U.inch).to(self.U.degR).magnitude + self.T_inf.magnitude) * self.U.degR)
        if T > self.T_sat:
            return self.T_sat
        else:
            return T


    def T_clad_init(self, z):
        """ Returns the clad temperature """
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
        return ((self.T_coolant(z).magnitude
               + (self.q_dprime*self.q_z(z)/self.hc).to(self.U.degR).magnitude) * self.U.degR)


    def T_clad(self, z):
        """ Returns the clad temperature """
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
        if self.method == 'thom' or self.method == 'jens-lottes':
            if z <= self.zn.magnitude:
                return ((self.T_coolant(z).magnitude
                       + (self.q_dprime*self.q_z(z)/self.hc).to(self.U.degR).magnitude) * self.U.degR)
            elif z > self.zn.magnitude and z <= self.zb.magnitude:
                return (self.T_clad_mixed(z))
            elif z > self.zb.magnitude:
                return (self.T_clad_zb(z))
        elif self.method.startswith('superposition'):
            if z >= 0 and z <= self.T_clad_sat_location().magnitude:
                return ((self.T_coolant(z).magnitude
                       + (self.q_dprime*self.q_z(z)/self.hc).to(self.U.degR).magnitude) * self.U.degR)
            elif z > self.T_clad_sat_location().magnitude:
                return (self.T_clad_superposition(z))
        elif self.method.startswith('chen'):
            return self.T_clad_chen(z)

    def T_sfuel(self, z):
        """ Returns the fuel surface temperature """
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
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


    def max_T_clad(self):
        """ Finds max T_clad temperature and location """
        zrange = np.linspace(0,self.height.magnitude, self.height.magnitude+1)
        temp = []
        for i in zrange:
            temp.append(self.T_clad(i))
        return max(temp), temp.index(max(temp)) *self.U.inch


    def max_T_fuel(self):
        """ Finds max fuel centerline temperature and location """
        zrange = np.linspace(0, self.height.magnitude, self.height.magnitude+1)
        temp = []
        for i in zrange:
            temp.append(self.T_fuel(i))
        return max(temp), temp.index(max(temp)) *self.U.inch


    def root_T_clad_sat(self, z):
        """ Function used for solving root for T_clad=T_sat"""
        return self.T_sat - self.T_clad_init(z)


    def T_clad_sat_location(self):
        """ Returns the location where T_clad = T_sat """
        return root(self.root_T_clad_sat, int(self.height.magnitude/3)).x[0] * self.U.inch


    @property
    def zeta(self):
        """ Returns zeta a correlation value """
        if self.method.endswith('jens-lottes'):
            self.m = 4
            return (np.exp(4*self.pressure.magnitude/900)/(60**4)
                    * self.U.Btu/self.U.hour/self.U.foot**2/self.U.degR**4)
        else:
            self.m = 2
            return (np.exp(2*self.pressure.magnitude/1260)/(72**2)
                    * self.U.Btu/self.U.hour/self.U.foot**2/self.U.degR**2)


    def root_zn(self, z):
        """ Function used for solving root for zn location """
        return (-self.q_dprime.to(self.U.Btu/self.U.hour/self.U.foot**2).magnitude *
            self.q_z(z) +
            (15.6*self.pressure.magnitude**1.156 * ((self.T_clad_init(z)-
            self.T_sat).magnitude)**(2.3/self.pressure.magnitude**.0234)))


    def find_zn(self):
        """ Finds the location of incipient boiling """
        return root(self.root_zn, int(self.T_clad_sat_location().magnitude+2)).x[0] * self.U.inch


    def root_zb(self, z):
        """ Returns root for finding zb """
        T_zb = self.T_sat + self.zeta**(-1/self.m)*(self.q_dprime*self.q_z(z)/10**6)**(1/self.m)
        T_zn = self.T_clad_init(self.zn)
        q_fc = self.hc*(T_zb-self.T_coolant(z))
        q_n = self.zeta*10**6*(T_zn-self.T_sat)**self.m
        q_dprime1 = self.q_dprime*self.q_z(z)
        return (q_fc*(1 + (q_dprime1/q_fc * (1 - q_n/(q_dprime1)))**2)**0.5
                - q_dprime1)


    def find_zb(self):
        """ Finds the boiling transition point """
        return root(self.root_zb, self.zb_guess).x[0] * self.U.inch


    def root_T_clad_mixed(self, T_clad, z):
        """ Root function for T_clad_mixed """
        q_fc = (self.hc*(T_clad*self.U.degR - self.T_coolant(z)))
        q_nb = (self.zeta*10**6*(T_clad*self.U.degR - self.T_sat)**self.m)
        q_n = (self.zeta*10**6*(self.T_clad_init(self.zn) - self.T_sat)**self.m)
        return (q_fc*(1 + (q_nb/q_fc * (1 - q_n/q_nb))**2)**0.5 - self.q_dprime
                * self.q_z(z))


    def T_clad_mixed(self, z):
        """ Finds T_clad of mixed region """
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
        return root(self.root_T_clad_mixed, 1200, args=(z)).x[0] * self.U.degR


    def T_clad_zb(self, z):
        """ Finds T_clad of zb region """
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
        return self.T_sat + self.zeta**(-1/self.m)*(self.q_dprime*self.q_z(z)/10**6)**(1/self.m)


    def root_T_clad_superposition(self, T_clad, z):
        """ Returns root T_clad by superposition approach """
        return (self.hc*(T_clad*self.U.degR-self.T_coolant(z))+self.h_NB(T_clad*self.U.degR)*
                (T_clad*self.U.degR-self.T_sat)) - self.q_dprime*self.q_z(z)


    def T_clad_superposition(self, z):
        """ Returns T_clad by superposition approach """
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
        return root(self.root_T_clad_superposition, 1200, args=(z)).x[0]*self.U.degR


    def root_T_coolant_sat(self, z):
        """ Function used for solving root for T_coolant=T_sat """
        return self.T_sat - self.T_coolant(z)


    def T_coolant_sat_location(self):
        """ Finding where T_sat=T_coolant """
        return root(self.root_T_coolant_sat, int(self.height.magnitude/2)).x[0] * self.U.inch


    def h_NB(self, T_clad):
        """ Returns h_NB for superposition approach """
        return self.zeta*10**6*(T_clad-self.T_sat)**(self.m-1)


    def h_2flux(self):
        """ Returns the nucleate boiling h_2flux """
        J=778*self.U.foot*self.U.lb/self.U.Btu
        top = self.k_f**.79*self.cp_f**.45*self.rho_f**.49*self.gc**.25
        bottom = self.sigma**0.5*self.mu_f**.29*self.h_fg**.24*self.rho_g**.24
        term_1 = ((self.h_fg*J)/(self.T_sat*self.v_fg))**.75
        return (.00122*((1*self.U.lbf)/(1*self.U.lb))**.75*(top/bottom)*term_1*10)


    def S(self, z):
        """ Returns S for chen """
        return (.9622-.5822*np.arctan(self.reynolds_lo(z)*self.F(z)**1.25/(6.18*10**4))).magnitude


    def reynolds_lo(self, z):
        """ Returns the reynolds lo value  """
        return self.G*(1-self.quality(z))*self.De/self.mu


    def quality(self, z):
        """ Returns the quality at a point """
        q = ((self.h_z(z)-self.h_f)/(self.h_fg)).to(self.U.dimensionless).magnitude
        if q <= 0:
            q = 0
        return q


    def abs_quality(self, z):
        """ Returns the absolute quality at a point """
        return ((self.h_z(z)-self.h_f)/(self.h_fg)).to(self.U.dimensionless).magnitude


    def quality_bounds(self, z, bound):
        """ Returns the valid quality bounds for root function """
        if bound == 'lower':
            return self.abs_quality(z) - (-.15)
        if bound == 'upper':
            return self.abs_quality(z) - (.15)

    def quality_range(self):
        """ Returns range of valid z positions for DNBR"""
        lower = int(root(self.quality_bounds, 50, args=('lower')).x[0])
        upper = int(root(self.quality_bounds, 150, args=('upper')).x[0])
        if upper >= self.height.magnitude:
            upper = int(self.height.magnitude)
        zrange = np.linspace(lower, upper, (upper-lower)+1)
        return zrange


    def h_z(self, z):
        """" Return h at a point """
        return (self.h_in + (1/(self.G*self.Ax*self.gamma)*np.pi*self.D_clad*self.q_dprime*
            quad(self.q_z, 0, z)[0]*self.U.inch))


    def martinelli(self, z):
        """ Returns martinelli for chen """
        return ((self.quality(z)/(1-self.quality(z)))**.9*(self.rho_f/self.rho_g)**.5*(self.mu_g/self.mu_f)**.1).magnitude


    def F(self, z):
        """ Returns factor F for chen """
        if self.martinelli(z) <= .1:
            return 1
        else:
            return 2.35*(self.martinelli(z)+.213)**.736


    def h_lo_dittus(self, z):
        """" Returns h_lo weismann for chen """
        h = .023*(self.G*(1-self.quality(z))*self.De/self.mu)**.8*(self.prandt)**0.4*(self.k_water/self.De)*self.F(z)
        return h.to(self.U.Btu/self.U.hour/self.U.foot**2/self.U.degR)


    def h_lo_weismann(self, z):
        """ Returns h_lo weismann for chen """
        h = (.042*self.pitch/self.D_clad-.024)*self.reynolds**.8*self.prandt**(1/3)*(self.k_water/(4*(self.pitch**2-np.pi*(self.D_clad**2/4))/(np.pi*self.D_clad)))*self.F(z)
        return h.to(self.U.Btu/self.U.hour/self.U.foot**2/self.U.degR)


    def root_chen(self, Tw, z):
        """ Returns root for chen """
        if self.method == 'chen-dittus':
            tmp = (self.S(z)*self.h2_flux*(Tw*self.U.degR-self.T_sat)**1.99).magnitude * (self.U.Btu/self.U.hour/self.U.foot**2)
            return self.h_lo_dittus(z)*(Tw*self.U.degR-self.T_coolant(z)) + tmp - self.q_dprime*self.q_z(z)
        elif self.method == 'chen-weismann':
            tmp = (self.S(z)*self.h2_flux*(Tw*self.U.degR-self.T_sat)**1.99).magnitude * (self.U.Btu/self.U.hour/self.U.foot**2)
            return self.h_lo_weismann(z)*(Tw*self.U.degR-self.T_coolant(z)) + tmp - self.q_dprime*self.q_z(z)

    def T_clad_chen(self, z):
        """ Returns the T_clad for chen """
        return root(self.root_chen, 1100, args=(z)).x[0]*self.U.degR


    def critical_heat_flux(self, z):
        """ Returns the critical heat flux at a position """
        x = self.abs_quality(z)
        return ((10**6*  (((2.022-.0004302*self.pressure.magnitude)+(.1722-.0000984*self.pressure.magnitude)*
                np.exp((18.177-.004129*self.pressure.magnitude)*x)) *
                ((.1484-1.596*x + .1729*x*abs(x))*self.G.magnitude/10**6+1.037)*
                (1.157-.869*x) * (.2664+.8357*np.exp(-3.151*self.De.magnitude))*
                (.8258+.000794*(self.h_f-self.h_in).magnitude)) ) * self.U.Btu/self.U.hour/self.U.foot**2)


    def critical_heat_flux_F(self, z):
        """ Returns the critical heat flux at a position """
        x = self.abs_quality(z)
        return ((10**6*  (((2.022-.0004302*self.pressure.magnitude)+(.1722-.0000984*self.pressure.magnitude)*
                np.exp((18.177-.004129*self.pressure.magnitude)*x)) *
                ((.1484-1.596*x + .1729*x*abs(x))*self.G.magnitude/10**6+1.037)*
                (1.157-.869*x) * (.2664+.8357*np.exp(-3.151*self.De.magnitude))*
                (.8258+.000794*(self.h_f-self.h_in).magnitude)) / self.F_2(z) ) * self.U.Btu/self.U.hour/self.U.foot**2)


    def int_F(self, z, c, L):
        """ Returns the integral function used in F_2 """
        return self.q_z(z)*np.exp(-c*(L-z))


    def find_l_EU(self, z):
        """ Returns l_EU at a point """
        return ((self.G*self.Ax*self.gamma*(self.h_z(z)-self.h_in))/(self.critical_heat_flux(z)*np.pi*self.D_clad)).to(self.U.inch)


    def F_2(self, z):
        """ Returns shape function for critical heat flux """
        c = (.44* ((1-self.abs_quality(z))**7.9)/((self.G.magnitude/10**6)**1.72))*self.U.inches**(-1)
        return ((  c/(self.q_z(z)*(1-np.exp(-c*self.find_l_EU(z)))) *
                quad(self.int_F, 0, z, args=(c.magnitude, z))[0]*self.U.inch  ).magnitude)


    def DNBR(self, z):
        """ Returns the DNBR ratio at a point """
        return (((self.critical_heat_flux_F(z)) / (self.q_dprime*self.q_z(z))).to(self.U.dimensionless)).magnitude


    def MDNBR(self):
        """ Returns the minimum DNBR value and location """
        temp = []
        for i in self.quality_range():
            temp.append(self.DNBR(i))
        return min(temp), (self.quality_range()[0]+temp.index(min(temp))) *self.U.inch


    def x_crit(self):
        """ Returns critical quality """
        Dh = ((self.D_clad).to(self.U.m)).magnitude
        De = ((self.De).to(self.U.m)).magnitude
        L_crit = 1
        return Dh/De*(self.a()*L_crit/(L_crit+self.b()))


    def crit_quality(self, z):
        """ Return quality from critical heat flux """
        return (((self.a_crit*(z*self.U.inch-self.h_0)) /
                (z*self.U.inch-self.h_0+self.b_crit)).to(self.U.dimensionless).magnitude)


    @property
    def a_crit(self):
        """ Returns a for critical problem """
        p_crit = ((self.p_crit).to(self.U.MPa)).magnitude
        p = ((self.pressure).to(self.U.MPa)).magnitude
        G = ((self.G).to(self.U.kg/self.U.m**2/self.U.s)).magnitude
        tmp = (1-p/p_crit)**3
        if G <= 3375 * tmp:
            return (1/(1+1.481*10**-4*tmp*G))*(self.D_clad/self.De)
        else:
            return (tmp**(1/3)/(G/1000)**(1/3))*(self.D_clad/self.De)

    @property
    def b_crit(self):
        """ Return b for citical problem """
        p_crit = ((self.p_crit).to(self.U.MPa)).magnitude
        p = ((self.pressure).to(self.U.MPa)).magnitude
        G = ((self.G).to(self.U.kg/self.U.m**2/self.U.s)).magnitude
        D = ((self.D_clad).to(self.U.m)).magnitude
        return (.199*(p_crit/p - 1)**(0.4)*G*D**1.4)*self.U.m

    # TODO
    def root_h0(self, h0):
        """ Returns root for h0 value for CISE correlation """
        left = (self.a_crit*(self.height-h0*self.U.inch))/(self.b_crit+(self.height-h0*self.U.inch))
        right1 = (self.h_f-self.h_in)/self.h_fg
        right2 = quad(self.q_z, h0, self.height.magnitude)[0]
        right3 = quad(self.q_z, 0, h0)[0]
        return left - right1*(right2/right3)


    def find_h0(self):
        """ Returns the h0 value for CISE correlation """
        return root(self.root_h0, self.h0_guess).x[0] * self.U.inch


    def q_crit(self):
        """ Returns the q_crit value for CISE correlation """
        top = self.G*self.Ax*self.gamma*(self.h_f-self.h_in)
        bottom = np.pi*self.D_clad*quad(self.q_z, 0, self.h_0.magnitude)[0]*self.U.inch
        return top/bottom


    def max_Q(self):
        """ Returns the max Q based on CPR """
        self.q_dprime = self.q_crit() / self.CPR
        return (((self.q_crit()*np.pi*self.n_rods*self.D_clad*self.height*self.shape)/
                (self.CPR*self.gamma*self.PF_power)).to(self.U.MW))


    def alpha_dix(self, z, type='critical'):
        """ Returns the Dix correlation function """
        if type == 'critical':
            b = (self.rho_g / self.rho_f)**(0.1)
            beta = self.crit_quality(z) / (self.crit_quality(z) + (1-self.crit_quality(z)) * (self.rho_g/self.rho_f))
            V_gj = 2.9 * (((self.sigma*self.gravity*(self.rho_f-self.rho_g)) / (self.rho_f**2))**(1/4))
            Co = beta * (1+((1/beta)-1)**b)
            return ( (1) / (Co*(1 + ((1-self.crit_quality(z))*self.rho_g)/(self.crit_quality(z)*self.rho_f)) + (self.rho_g*V_gj) / (self.G*self.crit_quality(z))) )
        else:
            b = (self.rho_g / self.rho_f)**(0.1)
            beta = self.abs_quality(z) / (self.abs_quality(z) + (1-self.abs_quality(z)) * (self.rho_g/self.rho_f))
            V_gj = 2.9 * (((self.sigma*self.gravity*(self.rho_f-self.rho_g)) / (self.rho_f**2))**(1/4))
            Co = beta * (1+((1/beta)-1)**b)
            return ( (1) / (Co*(1 + ((1-self.abs_quality(z))*self.rho_g)/(self.abs_quality(z)*self.rho_f)) + (self.rho_g*V_gj) / (self.G*self.abs_quality(z))) )


    def x_c1(self):
        """ Return quality (x_c1) for Hench Levy """
        return .273-.212*(np.tanh(3*self.G.magnitude/10**6))**2


    def x_c2(self):
        """ Return quality (x_c2) for Hench Levy """
        return 0.5-.269*(np.tanh(3*self.G.magnitude/10**6))**2+.0346*(np.tanh(2*self.G.magnitude/10**6))**2


    def HL(self, z):
        """ Returns Hench Levy correlation q'' """
        x1 = self.x_c1
        x2 = self.x_c2
        x = self.abs_quality(z)
        if x <= x1:
            return 10**6 * (self.U.Btu/self.U.hour/self.U.foot**2)
        elif x >= x1 and x <= x2:
            return (1.9-3.3*x-.7*(np.tanh(3*self.G.magnitude/10**6))**2)*10**6 *(self.U.Btu/self.U.hour/self.U.foot**2)
        else:
            return (.6-.7*x-.09*(np.tanh(2*self.G.magnitude/10**6))**2)*10**6 *(self.U.Btu/self.U.hour/self.U.foot**2)



    def HL_corrected(self, z):
        """ Returns the corrected Hench Levy q'' """
        return self.HL(z)*(1.1-.1*((self.pressure.magnitude-600)/400)**1.25)


    def CHF(self, z):
        """ Returns the CHF ratio at a point """
        return (((self.HL_corrected(z)) / (self.q_dprime*self.q_z(z))).to(self.U.dimensionless)).magnitude


    def MCHF(self):
        """ Returns the minimum DNBR value and location """
        temp = []
        zrange = np.linspace(0, self.height.magnitude, self.height.magnitude+1)
        for i in zrange:
            temp.append(self.CHF(i))
        return min(temp), temp.index(min(temp)) *self.U.inch


    def plot(self, coolant=False, clad=False, fuel=False, units=None, method=None):
        """ Returns a plot of the specified regions when set to True """
        zrange = np.linspace(0,self.height.magnitude, int(self.height.magnitude/2))
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




class void_fractions(object):
    def __init__(self, heat_flux, pressure, mass_flux, enthalpy, height, PF_axial,
                 pitch, D_rod, U, intervals, life):
        self.U = U
        self.heat_flux = heat_flux.to(self.U.Btu/self.U.hour/self.U.foot**2)
        self.height = height.to(self.U.inch)
        self.pitch = pitch.to(self.U.inch)
        self.PF_axial = PF_axial
        self.D_rod = D_rod.to(self.U.inch)
        self.pressure = pressure.to(self.U.psi)
        self.G = mass_flux.to(self.U.lb/(self.U.hour*self.U.feet**2))
        self.h1 = enthalpy.to(self.U.Btu/self.U.lb)
        self.ax = self.pitch**2-(np.pi/4)*self.D_rod**2
        self.De = (4*(self.ax))/(np.pi*self.D_rod)
        self.gravity = 9.81 * self.U.m/self.U.s**2
        self.intervals = intervals

        obj0 = ST(P=pressure.to(self.U.MPa).magnitude, x=0)
        self.rho_l = (1/obj0.v*self.U.kg/self.U.m**3).to(self.U.lb/self.U.foot**3)
        self.h_f = (obj0.h*self.U.kJ/self.U.kg).to(self.U.Btu/self.U.lb)

        obj1 = ST(P=pressure.to(self.U.MPa).magnitude, x=1)
        self.rho_g = (1/obj1.v*self.U.kg/self.U.m**3).to(self.U.lb/self.U.foot**3)
        self.h_g = (obj1.h*self.U.kJ/self.U.kg).to(self.U.Btu/self.U.lb)
        self.sigma = (obj1.sigma*self.U.N/self.U.m).to(self.U.lbf/self.U.feet)
        self.life = life

        self.h_fg = self.h_g - self.h_f
        self.inst = 0

        if life == 'MOC':
            """ Finds the lambda for middle of cycle """
            def q_z2(z, lambda_):
                return np.sin(np.pi * ((z + lambda_) / (self.height.magnitude + 2*lambda_)))
            def find_lambda(lambda_):
                return self.PF_axial - (self.height.magnitude/(quad(q_z2, 0, self.height.magnitude, args=(lambda_))[0]))
            self.lambda_ = root(find_lambda, 1).x[0] * self.U.inch
        elif life == 'BOC':
            """ Finds the lambda for beginning of cycle """
            def trial(x):
                return x*np.cos(x) + np.sin(x)
            x1 = root(trial, 2).x[0]
            hold = np.sin(x1) * x1
            def q_z2(z, lambda_):
                return (np.pi*(self.height.magnitude+lambda_-z) / (self.height.magnitude + 2*lambda_))*np.sin(np.pi*(self.height.magnitude+lambda_-z) / (self.height.magnitude + 2*lambda_))
            def find_lambda(lambda_):
                return self.PF_axial - ((hold*self.height.magnitude)/(quad(q_z2, 0, self.height.magnitude, args=(lambda_))[0]))
            self.lambda_ = root(find_lambda, 36).x[0] * self.U.inch


        def find_q_dprime(x1):
            """ Used for finding the q naught """
            return ((x1/self.height) * quad(self.q_dprime, 0, self.height.magnitude)[0]*self.U.inch) - self.heat_flux.magnitude

        self.q_dprime_avg = root(find_q_dprime, 150000).x[0] * (self.U.Btu/self.U.hour/self.U.foot**2)


    def q_dprime(self, z):
        """ Returns the heat flux depending on life cycle and height in core """
        if self.life == 'MOC':
            return np.sin((np.pi*(z*self.U.inch+self.lambda_))/(self.height+2*self.lambda_))
        elif self.life == 'BOC':
            return ((np.pi*(self.height+self.lambda_-z*self.U.inch) / (self.height + 2*self.lambda_))
                   * np.sin(np.pi*(self.height+self.lambda_-z*self.U.inch) / (self.height + 2*self.lambda_)))


    def vol_fraction(self,alpha, z):
        """ Function used for finding volume fraction based upon core location """
        return (alpha - ((1) / (1 + ((1-self.quality(z*self.U.inch))*self.rho_g)/(self.quality(z*self.U.inch)*self.rho_l) +
                ((1-alpha)*self.rho_g*self.vr(alpha)) / (self.G*self.quality(z*self.U.inch)) )))


    def vr(self, alpha):
        """ Returns the vr value based on which regime is being used """
        if self.use == 'Bubbly':
            T = ( (1.41/(1-alpha)) * ((self.gravity*self.sigma*(self.rho_l-self.rho_g)) / (self.rho_l**2))**(1/4) ).to(self.U.feet/self.U.s)
        elif self.use == 'Bubbly_Slug':
            bubble1 = ( (1.41/(1-.1)) * ((self.gravity*self.sigma*(self.rho_l-self.rho_g)) / (self.rho_l**2))**(1/4) ).to(self.U.feet/self.U.s)
            slug2 = ( (0.345/(1-.2)) * ((self.gravity*self.De*(self.rho_l-self.rho_g)) / (self.rho_l))**(1/2) ).to(self.U.feet/self.U.s)
            T = bubble1 + ((alpha-.1)/(0.1))*(slug2-bubble1)
        elif self.use == 'Slug':
            T = ( (0.345/(1-alpha)) * ((self.gravity*self.De*(self.rho_l-self.rho_g)) / (self.rho_l))**(1/2) ).to(self.U.feet/self.U.s)
        elif self.use == 'Slug_Annular':
            slug1 = ( (0.345/(1-.65)) * ((self.gravity*self.De*(self.rho_l-self.rho_g)) / (self.rho_l))**(1/2) ).to(self.U.feet/self.U.s)
            rho = (1-.85)*self.rho_l + .85*self.rho_g
            v = self.G / rho
            annular2 = ( (v) / ((self.rho_g*(76-75*.85)/(self.rho_l*np.sqrt(.85)))**(1/2) + (.85*self.rho_g)/(rho)) ).to(self.U.feet/self.U.s)
            T = slug1 + ((alpha-.65)/(0.2))*(annular2-slug1)
        elif self.use == 'Annular':
            rho = (1-alpha)*self.rho_l + alpha*self.rho_g
            v = self.G / rho
            T = ( (v) / (((self.rho_g*(76-75*alpha)/(self.rho_l*np.sqrt(alpha)))**(1/2) + (alpha*self.rho_g)/(rho)) )).to(self.U.feet/self.U.s)
        return T


    def alpha_g(self, z):
        """ Returns the Dix correlation function to be solved by a root function  """
        b = (self.rho_g / self.rho_l)**(0.1)
        beta = self.quality(z) / (self.quality(z) + (1-self.quality(z)) * (self.rho_g/self.rho_l))
        V_gj = 2.9 * (((self.sigma*self.gravity*(self.rho_l-self.rho_g)) / (self.rho_l**2))**(1/4))
        Co = beta * (1+((1/beta)-1)**b)
        return ( (1) / (Co*(1 + ((1-self.quality(z))*self.rho_g)/(self.quality(z)*self.rho_l)) + (self.rho_g*V_gj) / (self.G*self.quality(z))) )


    def quality(self, z):
        """ Returns the quality at a given height """
        try:
            z = z.to(self.U.inch).magnitude
        except:
            z = z
        x = ((self.q_dprime_avg*np.pi*self.D_rod*quad(self.q_dprime, 0, z)[0]*self.U.inch)/(self.G*self.ax) + self.h1 - self.h_f)/self.h_fg
        return x


    def find_alpha(self, z):
        """ Recursively solves for alpha based on the regimes within the TRAC map """
        z = z.to(self.U.inch).magnitude
        use = ['Bubbly', 'Bubbly_Slug', 'Slug', 'Slug_Annular', 'Annular']
        self.use = use[self.inst]
        vol_fraction = root(self.vol_fraction, .5, args=(z)).x[0]
        if vol_fraction >= self.intervals[self.inst]:
            self.inst += 1
            self.use = use[self.inst]
            vol_fraction = root(self.vol_fraction, .5, args=(z)).x[0]
        return vol_fraction, self.use


    def superficial_velocity(self, z):
        """ Returns superficial velocity based upon position """
        j_l = (self.G * (1 - self.quality(z))) / self.rho_l
        j_g = (self.G * (self.quality(z))) / self.rho_g
        return j_l, j_g


    def Hewitt_Roberts(self, z):
        """ Return value of HR regime plot """
        xaxis = (self.superficial_velocity(z)[0]**2 *
            self.rho_l).to(self.U.kg/self.U.s**2/self.U.m)
        yaxis = (self.superficial_velocity(z)[1]**2 *
            self.rho_g).to(self.U.kg/self.U.s**2/self.U.m)
        return xaxis, yaxis


    def solver_for_HR(self, z, line):
        """ Returns intersection function for HR plot """
        if line == 'first':
            return (self.Hewitt_Roberts(z)[1]
                - 50 * (self.U.kg/self.U.s**2/self.U.m))
        if line == 'second':
            return (self.Hewitt_Roberts(z)[1]
                - 100 * (self.U.kg/self.U.s**2/self.U.m))


    def find_z_for_Hline(self):
        """ Find the axial height of HR intersection """
        z1 = root(self.solver_for_HR, 50, args=('first')).x[0]
        z2 = root(self.solver_for_HR, 100, args=('second')).x[0]
        return z1, z2


    def find_0_quality(self):
        """ Finds the axial point of zero quality """
        return root(self.quality, 30).x[0]


    def plot(self):
        """" Plots the functions derived above """
        x = self.find_0_quality()
        tmp, tmp2, tmp3, xlist = [], [], [], []
        for i in range(int(x)+1, self.height.magnitude):
            tmp.append(self.find_alpha(i*self.U.inch)[0])
            tmp2.append (self.find_alpha(i*self.U.inch)[1])
            tmp3.append(self.alpha_g(i*self.U.inch))
            xlist.append(i)
        plt.figure(figsize=(10,8))
        for type in np.unique(tmp2):
            beg = tmp2.index(type)
            end = len(tmp2)-tmp2[::-1].index(type)
            plt.plot(xlist[beg:end],tmp[beg:end], label=type)
        plt.plot(xlist, tmp3, label='Dix')
        plt.plot(np.linspace(0,x,50), np.zeros(50))
        self.inst = 0
        plt.axhspan(0, self.find_alpha(self.find_z_for_Hline()[0]
            *self.U.inch)[0], color='red', alpha=.1, label='HR Bubble Slug')
        plt.axhspan(self.find_alpha(self.find_z_for_Hline()[0]*self.U.inch)[0],
            self.find_alpha(self.find_z_for_Hline()[1]
            *self.U.inch)[0], color='blue', alpha=.1, label='HR Churn')
        plt.axhspan(self.find_alpha(self.find_z_for_Hline()[1]
            *self.U.inch)[0], max(tmp)+.05, color='green',
            alpha=.1, label='HR Wispy-Annular')
        plt.legend()
        plt.xlabel('Core Height [inches]')
        plt.ylabel('Volume Fraction')
        plt.show()

    def plot_Hewitt_Roberts(self):
        """ Plot on same axes as Hewitt and Roberts plot """
        x = self.find_0_quality()
        tmp1, tmp2 = [], []
        for i in range(int(x)+1, self.height.magnitude):
            tmp1.append(self.Hewitt_Roberts(i)[0].magnitude)
            tmp2.append(self.Hewitt_Roberts(i)[1].magnitude)
        plt.figure(figsize=(10,8))
        plt.plot(tmp1, tmp2)
        plt.xlabel(r'$\rho_L J_L [\frac{kg}{s^2m}]$')
        plt.ylabel(r'$\rho_G J_G [\frac{kg}{s^2m}]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(.1, 10**6)
        plt.ylim(.01, 10**6)
        plt.show()
