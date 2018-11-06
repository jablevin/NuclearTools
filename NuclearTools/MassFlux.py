import pint
import numpy as np
import scipy
from scipy.optimize import root
from scipy.integrate import quad
import matplotlib.pyplot as plt
from iapws import IAPWS97 as ST
from scipy.misc import derivative
from scipy.optimize import fsolve


class steam_generator(object):
    def __init__(self, m, T_hl, T_cl, A_ht, n_tubes, D, wall_th,
                L, radius_max, radius_min, plate_th, inlet_k,
                exit_k, eq_long, eq_short, U):

        #Inputs
        self.m = m
        self.T_hl = T_hl
        self.T_cl = T_cl
        self.A_ht = A_ht
        self.n_tubes = n_tubes
        self.D = D
        self.wall_th = wall_th
        self.L = L
        self.radius_max = radius_max
        self.radius_min = radius_min
        self.plate_th = plate_th
        self.inlet_k = inlet_k
        self.exit_k = exit_k
        self.eq_long = eq_long
        self.eq_short = eq_short
        self.U = U

        #Start Calculations
        self.Di = D - 2*wall_th
        self.G = m / (n_tubes * np.pi/4 * self.Di**2)
        self.L_long = 2*self.L + np.pi * self.radius_max
        self.L_short = 2*self.L + np.pi * self.radius_min
        self.L_avg = (self.L_long + self.L_short) / 2
        self.L_tube = (A_ht / (n_tubes*np.pi*D)).to(self.U.foot)

        T_avg = (T_hl+T_cl)/2
        obj = ST(T=T_avg.to(self.U.degK).magnitude, x=0)
        obj = ST(T=T_avg.to(self.U.degK).magnitude, P=(2250*self.U.psi).to(self.U.MPa).magnitude)
        self.rho = (1/(obj.v * (self.U.m**3/self.U.kg))).to(self.U.lb/self.U.foot**3)
        self.mu = (obj.mu * self.U.Pa * self.U.s).to(self.U.lb/(self.U.foot*self.U.hour))
        self.epsilon = .000005*self.U.foot
        self.Re = ((self.G * self.Di) / self.mu).to(self.U.dimensionless).magnitude
        self.friction = root(self.root_friction, .01, args=(self.epsilon,self.Di,self.Re)).x[0]

        self.dP_plate = (self.inlet_k + self.friction*self.plate_th/self.Di) * self.G**2/(2*self.rho) + self.G**2/(2*self.rho)
        self.dP_loss = self.friction * (2*self.L/self.Di + (self.eq_long+self.eq_short)/2) * self.G**2/(2*self.rho)
        self.dP_exit = (self.exit_k + self.friction*self.plate_th/self.Di) * self.G**2/(2*self.rho)
        self.total_dp = (self.dP_plate + self.dP_loss + self.dP_exit).to(U.psi)

        self.long_k = (self.friction * (self.L_long/self.Di +  eq_long)).to(self.U.dimensionless).magnitude
        self.short_k = (self.friction * (self.L_short/self.Di + eq_short)).to(self.U.dimensionless).magnitude

        self.v_avg = self.G / self.rho
        self.G_long = root(self.long, 10**6).x[0] * (self.U.lb/self.U.foot**2/self.U.hour)
        self.G_short = root(self.short, 10**6).x[0] * (self.U.lb/self.U.foot**2/self.U.hour)
        self.v_long = self.G_long / self.rho
        self.v_short = self.G_short / self.rho

    def root_friction(self, f, epsilon, D, Re):
        return 1.14-2*np.log10(epsilon/D+(9.35)/(Re*np.sqrt(f)))-1/np.sqrt(f)

    def long(self, G):
        return self.friction * (2*self.L/self.Di + self.eq_long) * (G*(self.U.lb/self.U.foot**2/self.U.hour))**2/(2*self.rho) - self.dP_loss

    def short(self, G):
        return self.friction * (2*self.L/self.Di + self.eq_short) * (G*(self.U.lb/self.U.foot**2/self.U.hour))**2/(2*self.rho) - self.dP_loss



class core_pressure(object):
    def __init__(self, pitch, D_clad, n_rods, height, pressure, n_grids,
                k_grid, core_height, k_core_in, k_core_out, v_ID,
                b_OD, L_d, k_d, L_hl, D_hl, HL_LD, k_hl_in, k_hl_out,
                k_sg_in, k_sg_out, SG_LD, D_sg, SG_th, n_tubes,
                A_total, L_cl, D_cl, k_cl_in, k_cl_out, CL_LD,
                T_in, T_out, m, U, loops):

        self.pitch, self.D_clad, self.n_rods, self.pressure, self.height, self.n_grids = pitch, D_clad, n_rods, height, pressure, n_grids
        self.k_grid, self.core_height, self.k_core_in, self.k_core_out, self.v_ID = k_grid, core_height, k_core_in, k_core_out, v_ID
        self.b_OD, self.L_d, self.k_d, self.L_hl, self.D_hl = b_OD, L_d, k_d, L_hl, D_hl
        self.k_hl_in, self.k_hl_out, self.HL_LD = k_hl_in, k_hl_out, HL_LD
        self.k_sg_in, self.k_sg_out, self.SG_LD = k_sg_in, k_sg_out, SG_LD
        self.D_sg, self.SG_th, self.n_tubes = D_sg, SG_th, n_tubes
        self.A_total, self.L_cl, self.D_cl, self.k_cl_in = A_total, L_cl, D_cl, k_cl_in
        self.k_cl_out, self.CL_LD = k_cl_out, CL_LD
        self.T_in, self.T_out, self.m, self.U, self.loops = T_in, T_out, m, U, loops

        self.U = U
        self.g = 32.2 * U.foot / U.s**2
        g = 32.2 * U.foot / U.s**2

        obj = ST(T=((T_in+T_out)/2).to(self.U.degK).magnitude, P=pressure.to(self.U.MPa).magnitude)
        self.rho = (1/(obj.v * (self.U.m**3/self.U.kg))).to(self.U.lb/self.U.foot**3)
        self.mu = (obj.mu * self.U.Pa * self.U.s).to(self.U.lb/(self.U.foot*self.U.hour))

        self.epsilon1 = .000005*self.U.foot
        self.epsilon2 = .00015*self.U.foot

        # Total Pressure Drop
        self.P_total = (self.P_sg(self.m) + self.P_hot_leg(self.m) + self.P_cold_leg(self.m) +
            self.P_core(self.m) + self.P_downcomer(self.m) + self.rho*self.g*self.L_d - self.rho*self.g*self.core_height)
        self.G_cl_stable = self.G_cl
        # Pump Work
        self.work = (self.m/self.loops) * (self.P_total/self.rho)/.75
        self.G_core_initial = self.G_core
        self.unit1 = 1 * self.U.lb / self.U.hour / self.U.foot**2
        self.G_values = fsolve(self.function, (10**7, 10**7, 10**6))

    def P_core(self, m=None, G=False):
        self.A_core = self.n_rods*(self.pitch**2-np.pi*self.D_clad**2/4)
        self.De_core = (4*(self.pitch**2-np.pi*self.D_clad**2/4)) / (np.pi * self.D_clad)
        if not G:
            self.G_core = m / self.A_core
        else:
            self.G_core = G
        self.Re_core = (self.G_core * self.De_core / self.mu).to(self.U.dimensionless)
        self.fr_core = root(self.root_friction, .01, args=(self.epsilon1,self.De_core,self.Re_core)).x[0]
        return ((self.fr_core*self.core_height/self.De_core + self.k_core_in + self.k_core_out + self.n_grids*self.k_grid) *
                        self.G_core**2/(2*self.rho) + self.rho*self.g*self.core_height)

    def P_downcomer(self, m=None, G=False):
        self.A_down = np.pi/4*(self.v_ID**2 - self.b_OD**2)
        self.De_down = self.v_ID - self.b_OD
        if not G:
            self.G_down = m / self.A_down
        else:
            self.G_down = G
        self.Re_downcomer = self.G_down * self.De_down / self.mu
        self.fr_down = root(self.root_friction, .01, args=(self.epsilon2,self.De_down,self.Re_downcomer)).x[0]
        return (self.fr_down*self.L_d/self.De_down + self.k_d) * self.G_down**2/(2*self.rho) - self.rho*self.g*self.L_d

    def P_hot_leg(self, m=None, G=False):
        self.A_hl = np.pi*self.D_hl**2/4
        if not G:
            self.G_hl = m / (self.loops*self.A_hl)
        else:
            self.G_hl = G
        self.Re_hot_leg = self.G_hl * self.D_hl / self.mu
        self.fr_hl = root(self.root_friction, .01, args=(self.epsilon2,self.D_hl,self.Re_hot_leg)).x[0]
        return (self.fr_hl*(self.L_hl/self.D_hl + self.HL_LD) + self.k_hl_in + self.k_hl_out) * self.G_hl**2/(2*self.rho)

    def P_sg(self, m=None, G=False):
        self.De_sg = self.D_sg-2*self.SG_th
        self.A_sg = np.pi*self.De_sg**2/4
        self.L_sg = (self.A_total/(self.n_tubes*np.pi*self.D_sg).to(self.U.foot))
        if not G:
            self.G_sg = m / (4*self.n_tubes*self.A_sg)
        else:
            self.G_sg = G
        self.Re_sg = self.G_sg * self.De_sg/self.mu
        self.fr_sg = root(self.root_friction, .01, args=(self.epsilon1,self.De_sg,self.Re_sg)).x[0]
        return (self.fr_sg*(self.L_sg/(self.De_sg) + self.SG_LD) + self.k_sg_in + self.k_sg_out) * self.G_sg**2/(2*self.rho)

    def P_cold_leg(self, m=None, G=False):
        self.A_cl = np.pi*self.D_cl**2/4
        if not G:
            self.G_cl = m / (self.loops*self.A_cl)
        else:
            self.G_cl = G
        self.Re_cl = self.G_cl * self.D_cl/self.mu
        self.fr_cl = root(self.root_friction, .01, args=(self.epsilon2,self.D_cl,self.Re_cl)).x[0]
        return (self.fr_cl*(self.L_cl/self.D_cl + self.CL_LD) + self.k_cl_in + self.k_cl_out) * self.G_cl**2/(2*self.rho)

    def root_friction(self, f, epsilon, D, Re):
        return 1.14-2*np.log10(epsilon/D+(9.35)/(Re*np.sqrt(f)))-1/np.sqrt(f)


    def P_pump(self, G):
        return (1.094+.089*(G/self.G_cl_stable)-.183*(G/self.G_cl_stable)**2) * self.P_total

    def function(self, G):
        G_hl1, G_hl2, G_core = G
        G_hl1, G_hl2, G_core = G_hl1*self.unit1, G_hl2*self.unit1, G_core*self.unit1
        ratio = self.A_hl/self.A_cl
        ratio2 = self.A_hl/(self.A_sg*self.n_tubes)
        ratio3 = self.A_core/self.A_down
        return ((self.P_core(G=G_core) + self.P_downcomer(G=G_core*ratio3) + self.P_hot_leg(G=G_hl1) + self.P_sg(G=(G_hl1*ratio2/.9)) +
                    self.P_cold_leg(G=G_hl1*ratio) - self.P_pump(G=G_hl1*ratio)+ self.rho*self.g*self.L_d - self.rho*self.g*self.core_height).to(self.U.psi).magnitude,

                    (self.P_core(G=G_core) + self.P_downcomer(G=G_core*ratio3) + self.P_hot_leg(G=G_hl2) + self.P_sg(G=(G_hl2*ratio2)) +
                    self.P_cold_leg(G=G_hl2*ratio) - self.P_pump(G=G_hl2*ratio)+ self.rho*self.g*self.L_d - self.rho*self.g*self.core_height).to(self.U.psi).magnitude,

                    (G_hl1*self.A_hl + (self.loops-1)*G_hl2*self.A_hl - G_core*self.A_core).to(self.U.pound/self.U.hour).magnitude)
