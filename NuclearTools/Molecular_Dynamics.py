from ctypes import *
import numpy as np
import matplotlib.pyplot as plt

class MD(object):
    def __init__(self=None, deltaT=None, density=None, initUcellx=None, initUcelly=None, stepAvg=None,
                stepEquil=None, stepLimit=None, temperature=None, limitVel=None, rangeVel=None,
                sizeHistVel=100, stepVel=None, randSeedP=None, printout=False, finished=False):
        if finished:
            pass
        else:
            so_file = "build/lib.win-amd64-3.7/MD.cp37-win_amd64.pyd"
            lib = CDLL(so_file)
            lib.PyInit_MD.argtypes = [c_double, c_double, c_int, c_int, c_int, c_int,
                c_int, c_double, c_int, c_double, c_int, c_int]
            if printout:
                printout = 1
            else:
                printout = 0

            lib.PyInit_MD(deltaT,
                          density,
                          initUcellx,
                          initUcelly,
                          stepAvg,
                          stepEquil,
                          stepLimit,
                          temperature,
                          limitVel,
                          rangeVel,
                          sizeHistVel,
                          stepVel,
                          randSeedP,
                          printout)

        with open('output.txt', 'r') as search:
            lines = search.readlines()
            temp, temp2, temp3 = -np.ones((len(lines), 2)), -np.ones((2, len(lines))), -np.ones((len(lines), 10))
            j, k, m = 0, 0, 0
            for i in range(2, len(lines)):
                if lines[i].startswith('hfun'):
                    temp2[0][k], temp2[1][k] = lines[i][8:14], lines[i][17:23]
                    k += 1
                elif lines[i].startswith('Summary'):
                    temp3[m] = (lines[i][8:]).split()
                    m += 1
                try:
                    temp[j] = lines[i].split()
                    j += 1
                except:
                    pass
            t = np.where(temp == -1)[0][0]
            t2 = np.where(temp2[0] == -1)[0][0]
            t3 = np.where(temp3 == -1)[0][0]
            temp = np.array(temp[:t, :t])
            self.hfun = temp2[:t2, :t2]

            self.stepcount = temp3[:t3, 0]
            self.time = temp3[:t3, 1]
            self.momentum = temp3[:t3, 2]
            self.totenergy = temp3[:t3, 3]
            self.sig_totenergy = temp3[:t3, 4]
            self.kinenergy = temp3[:t3, 5]
            self.sig_kinenergy = temp3[:t3, 6]
            self.pressure = temp3[:t3, 7]
            self.sig_pressure = temp3[:t3, 8]
            self.potenergy = temp3[:t3, 3] - temp3[:t3, 5]
            self.temperature = 2/(3*1.38064E-23) * self.kinenergy

            self.hist = np.ones((t, 2, sizeHistVel))
            counter = t // sizeHistVel
            j = 0
            for k in range(counter):
                self.hist[j] = (temp[k*sizeHistVel:(k+1)*sizeHistVel,  :]).T
                j += 1


    def plot_prob_vs_vel(self, increments=[0, 1, 2, 5, 10]):
        plt.figure(figsize=(12,8))
        for i in increments:
            plt.plot(self.hist[i][0], self.hist[i][1], label=i)
        plt.xlabel('Velocity')
        plt.ylabel('Probability')
        plt.legend()
        plt.show()


    def plot_hfun_vs_time(self, low_r=0, up_r=10):
        plt.figure(figsize=(12,8))
        plt.plot(self.hfun[0][low_r:up_r], self.hfun[1][low_r:up_r])
        plt.ylabel('h-function')
        plt.xlabel('Time')
        plt.show()



    def fplot(self, property):
        values = {'Kinetic Energy': self.kinenergy,
                  'Potential Energy': self.potenergy,
                  'Total Energy': self.totenergy,
                  'Momentum': self.momentum,
                  'Temperature': self.temperature,
                  "Pressure": self.pressure}
        plt.figure(figsize=(12,8))
        plt.plot(self.time, values[property])
        plt.ylabel(property)
        plt.xlabel('Time')
        plt.show()
