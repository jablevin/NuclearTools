import pint
import numpy as np
import scipy
from scipy.optimize import root
from scipy.integrate import quad
import matplotlib.pyplot as plt
from iapws import IAPWS97 as ST

class reactor_temperatures(object):
    def __init__(self, power, height, pitch, T_inf, PF_power, PF_axial, D_clad, c_thick,
                 k_c, n_rods, h_g, pressure, mass_flux, percent, cp, U, T_sat):
        self.U = U
        self.power = power.to(self.U.MW)
        self.height = height.to(self.U.inch)
        self.pitch = pitch.to(self.U.inch)
        self.T_inf = T_inf.to(self.U.degR)
        self.PF_power = PF_power
        self.PF_axial = PF_axial
        self.D_clad = D_clad.to(self.U.inch)
        self.c_thick = c_thick.to(self.U.inch)
        D_pellet = D_clad-2*c_thick
        self.D_pellet = D_pellet.to(self.U.inch)
        self.k_c = k_c.to(self.U.Btu/(self.U.hour*self.U.feet*self.U.rankine))
        self.n_rods = n_rods
        self.h_g = h_g.to(self.U.Btu/(self.U.hour*self.U.feet**2*self.U.rankine))
        self.pressure = pressure.to(self.U.psi)
        self.mass_flux = mass_flux.to(self.U.lb/(self.U.hour*self.U.feet**2))
        self.percent = percent
        self.cp = cp.to(self.U.Btu/(self.U.lb*self.U.degR))
        self.T_sat = T_sat.to(self.U.degR)

        self.q_dprime_avg = ((percent * power * PF_axial) / (n_rods * np.pi * D_clad * height)).to(self.U.lb/self.U.second**3)
        self.q_dprime_max = self.q_dprime_avg * PF_power / PF_axial

        T_holding = (T_inf + T_sat) / 2
        obj = ST(P=pressure.to(self.U.MPa).magnitude, T=T_holding.to(self.U.degK).magnitude)

        self.mu = obj.mu * self.U.Pa * self.U.s
        self.k_water = obj.k * self.U.W / (self.U.m * self.U.degK)

        self.A1 = pitch**2 - np.pi*(D_clad**2)/4
        self.De = (4*(pitch**2 - np.pi*(D_clad**2)/4)) / (np.pi * D_clad)

        self.hc = (.042*pitch/D_clad-.024)*(mass_flux*self.De/self.mu)**0.8*.93**(1/3)*(self.k_water/(4*(pitch**2-np.pi*(D_clad**2/4))/(np.pi*D_clad)))
        self.hc = self.hc.to(self.U.Btu/(self.U.hour*self.U.foot**2*self.U.degR))

        def find_lambda(self):
            """Finds the """
            def q_z2(z, lambda_):
                return np.sin(np.pi * ((z + lambda_) / (self.height.magnitude + 2*lambda_)))
            def find_lambda(lambda_):
                return self.PF_axial - (self.height.magnitude/(quad(q_z2, 0, self.height.magnitude, args=(lambda_))[0]))

        self.lambda_ = root(find_lambda, 1).x[0] * self.U.inch


    def q_z(self, z):
        return np.sin((np.pi*(z*self.U.inch+self.lambda_))/(self.height+2*self.lambda_))


    def conduct_int(self, T):
        return 3978.1/(692.6+T) + 6.02366*10**(-12) * (T+460)**3


    def fuel_root(self, T, T_sfuel, z, q_dprime):
        return (quad(self.conduct_int, T_sfuel, T)[0]*self.U.Btu/(self.U.foot*self.U.hour) -
                ((q_dprime*self.q_z(z)*2*self.D_clad/2/(self.D_pellet/2)**2)*(self.D_pellet/2)**2/4))


    def T_coolant(self, z, q_dprime):
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
        return (((1/(self.mass_flux*self.A1*self.cp)*np.pi*self.D_clad*q_dprime*quad(self.q_z, 0, z,
                )[0]*self.U.inch).to(self.U.degR).magnitude + self.T_inf.magnitude) * self.U.degR)


    def T_clad(self, z, q_dprime):
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
        return ((self.T_coolant(z, q_dprime).magnitude
               + (q_dprime*self.q_z(z)/self.hc).to(self.U.degR).magnitude) * self.U.degR)


    def T_sfuel(self, z, q_dprime):
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
        return ((self.T_clad(z, q_dprime).magnitude
                + (q_dprime*self.q_z(z) * self.D_clad/2 *
                (1/(self.h_g*self.D_pellet/2) + 1/self.k_c*np.log(self.D_clad/self.D_pellet))).to(self.U.degR).magnitude) * self.U.degR)


    def T_fuel(self, z, q_dprime):
        try:
            z = z.to(self.U.inch).magnitude
        except:
            pass
        return (((root(self.fuel_root, 2000, args=(self.T_sfuel(z, q_dprime).to(self.U.degF).magnitude, z, q_dprime)).x[0])+458.67) * self.U.degR)


    def max_clad_location(self):
        def max_clad(z):
            init_1 = (np.pi*self.D_clad)/(self.mass_flux*self.A1*self.cp)*np.sin(np.pi*(z*self.U.inch+self.lambda_)/(self.height+2*self.lambda_))
            init_2 = np.pi/(self.hc*(self.height+2*self.lambda_))*np.cos(np.pi*(z*self.U.inch+self.lambda_)/(self.height+2*self.lambda_))
            return init_1 + init_2
        self.z_cladmax = root(max_clad, 70).x[0] * self.U.inch
        return self.z_cladmax


    def max_T_clad(self, q_dprime):
        return self.T_clad(self.max_clad_location().magnitude, q_dprime)


    def max_PF_boiling(self, q_dprime):
        self.max_clad_location()
        def func(PF_max):
            q_dprime_max = ((self.percent * self.power * PF_max) / (self.n_rods * np.pi * self.D_clad * self.height))
            init_1 = self.T_inf
            init_2 = (q_dprime_max * (1/(self.mass_flux*self.A1*self.cp) * np.pi * self.D_clad * quad(self.q_z, 0, self.z_cladmax.magnitude)[0]*self.U.inch)).to(self.U.degR)
            init_3 = (q_dprime_max * self.q_z(self.z_cladmax.magnitude) / self.hc).to(self.U.degR)
            return init_1 + init_2 + init_3 - self.T_sat
        temp = root(func, 1.5).x[0]
        return temp


    def plot(self, q_dprime, coolant=False, clad=False, fuel=False, units=None):
        zrange = np.linspace(0,self.height.magnitude, self.height.magnitude+1)
        T_coolant, T_clad, T_fuel = [], [], []

        if units == None:
            units = self.U.degF

        if units == self.U.degK:
            unit_string = str(units)[-6]
        else:
            unit_string = str(units)[-1]

        for z in zrange:
            if coolant:
                T_coolant.append(self.T_coolant(z, q_dprime).to(units).magnitude)
            if clad:
                T_clad.append(self.T_clad(z, q_dprime).to(units).magnitude)
            if fuel:
                T_fuel.append(self.T_fuel(z, q_dprime).to(units).magnitude)

        plt.figure(figsize=(10,6))
        if coolant:
            plt.plot(zrange, T_coolant, label='Coolant')
        if clad:
            plt.plot(zrange, T_clad, label='Clad')
        if fuel:
            plt.plot(zrange, T_fuel, label='Fuel')
        plt.legend()
        plt.ylabel('Temperature [' + unit_string + ']')
        plt.xlabel('Rod Height [inches]')
        plt.show()
