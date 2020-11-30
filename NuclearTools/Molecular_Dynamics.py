import numpy as np
import matplotlib.pyplot as plt
from NuclearTools import MD2D, MD3D
from abc import ABC, abstractmethod



class BaseFunctions(ABC):
    @abstractmethod
    def plot_prob_vs_vel(self, increments=[0, 1, 2, 5, 10], output_list=None):
        if output_list == None:
            output_list = list(self.outputs.keys())
        plt.figure(figsize=(12,8))
        for i in increments:
            for method in output_list:
                plt.plot(self.outputs[method].hist[i][0], self.outputs[method].hist[i][1], label=i)
        plt.xlabel('Velocity', fontsize=14)
        plt.ylabel('Probability', fontsize=14)
        plt.legend()
        plt.tight_layout()



    @abstractmethod
    def plot_hfun_vs_time(self, low_r=0, up_r=10, output_list=None):
        if output_list == None:
            output_list = list(self.outputs.keys())
        plt.figure(figsize=(12,8))
        for method in output_list:
            plt.plot(self.outputs[method].hfun[0][low_r:up_r], self.outputs[method].hfun[1][low_r:up_r])
        plt.ylabel('h-function', fontsize=14)
        plt.xlabel('Time', fontsize=14)
        plt.tight_layout()



    @abstractmethod
    def fplot(self, property, Lmean=True, output_list=None):
        if output_list == None:
            output_list = list(self.outputs.keys())
        plt.figure(figsize=(12,8))
        for method in output_list:
            plt.plot(self.outputs[method].time, self.outputs[method].values[property])
        if Lmean:
            for method in output_list:
                plt.axhline(np.mean(self.outputs[method].values[property]), color='crimson', label='Mean')
        plt.ylabel(property, fontsize=14)
        plt.xlabel('Time', fontsize=14)
        if Lmean:
            plt.legend()
        plt.tight_layout()



    @abstractmethod
    def fplot_mult(self, properties, output_list=None):
        if output_list == None:
            output_list = list(self.outputs.keys())
        plt.figure(figsize=(12,8))
        for type in properties:
            for method in output_list:
                plt.plot(self.outputs[method].time, self.outputs[method].values[type], label=type+' - '+method)
        plt.ylabel('Properties', fontsize=14)
        plt.xlabel('Time', fontsize=14)
        plt.legend()
        plt.tight_layout()



    @abstractmethod
    def fplot_ravg(self, properties, output_list=None):
        m_avg = {}
        if output_list == None:
            output_list = list(self.outputs.keys())
        for method in output_list:
            m_avg[method] = {}
            for name in list(self.outputs[method].values.keys()):
                m_avg[method][name] = []
                for N in range(1,len(self.outputs[method].values[name])):
                    m_avg[method][name].append(np.mean(self.outputs[method].values[name][0:N]))
        plt.figure(figsize=(12,8))
        for type in properties:
            for method in output_list:
                plt.plot(self.outputs[method].time[0:-1], m_avg[method][type], label=type+' - '+method)
        plt.ylabel('Properties', fontsize=14)
        plt.xlabel('Time', fontsize=14)
        plt.legend()
        plt.tight_layout()



class Reader(ABC):
    def __init__(self, outfile='output.txt'):
        with open(outfile, 'r') as search:
            lines = search.readlines()
            temp, temp2, temp3 = -np.ones((len(lines), 2)), -np.ones((2, len(lines))), -np.ones((len(lines), 11))
            j, k, m = 0, 0, 0
            keeper = []
            for i in range(len(lines)):
                if lines[i].startswith('hfun'):
                    temp2[0][k], temp2[1][k] = lines[i][8:14], lines[i][17:23]
                    keeper.append(i)
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
            vvsum = temp3[:t3, 10]
            self.temperature = 2 / 3 * np.array(self.kinenergy)
            self.sig_temperature = 2 / 3 * np.array(self.sig_kinenergy)
            sizeHistVel = np.min(keeper)
            self.hist = np.ones((t, 2, sizeHistVel))
            counter = t // sizeHistVel
            j = 0
            for k in range(counter):
                self.hist[j] = (temp[k*sizeHistVel:(k+1)*sizeHistVel,  :]).T
                j += 1

            self.values = {'Kinetic Energy': self.kinenergy,
                      'Potential Energy': self.potenergy,
                      'Total Energy': self.totenergy,
                      'Momentum': self.momentum,
                      'Temperature': self.temperature,
                      'Pressure': self.pressure,
                      'Total Energy stdv': self.sig_totenergy,
                      'Kinetic Energy stdv': self.sig_kinenergy,
                      'Pressure stdv': self.sig_pressure,
                      'Temperature stdv': self.sig_temperature}




class Simulate2D(BaseFunctions):
    def __init__(self, deltaT=None, density=None, initUcellx=None, initUcelly=None, stepAvg=None,
                stepEquil=None, stepLimit=None, temperature=None, limitVel=None, rangeVel=None,
                sizeHistVel=100, stepVel=None, randSeedP=None, printout=False, outfile='output.txt'):
        if printout == True or printout == 1:
            pout = 1
        else:
            pout = 0
        if not outfile.endswith(tuple(['.txt'])):
            outfile += '.txt'
        with open('input.txt', 'w') as in_file:
            in_file.write(str(deltaT) + '-----DeltaT\n')
            in_file.write(str(density) + '-----density\n')
            in_file.write(str(initUcellx) + '-----initUcellx\n')
            in_file.write(str(initUcelly) + '-----initUcelly\n')
            in_file.write(str(stepAvg) + '-----stepAvg\n')
            in_file.write(str(stepEquil) + '-----stepEquil\n')
            in_file.write(str(stepLimit) + '-----stepLimit\n')
            in_file.write(str(temperature) + '-----temperature\n')
            in_file.write(str(limitVel) + '-----limitVel\n')
            in_file.write(str(rangeVel) + '-----rangeVel\n')
            in_file.write(str(sizeHistVel) + '-----sizeHistVel\n')
            in_file.write(str(stepVel) + '-----stepVel\n')
            in_file.write(str(randSeedP) + '-----randSeedP\n')
            in_file.write(str(pout) + '-----printout\n')
            in_file.write(str(outfile) + '\n')
        MD2D.MD2D('input.txt')
        self.outputs = {}
        self.outputs[outfile] = Reader(outfile=outfile)


    def plot_prob_vs_vel(self, increments=[0, 1, 2, 5, 10]):
        super().plot_prob_vs_vel(increments)


    def plot_hfun_vs_time(self, low_r=0, up_r=10):
        super().plot_hfun_vs_time(low_r, up_r)


    def fplot(self, property, Lmean=True):
        super().fplot(property, Lmean)


    def fplot_mult(self, properties):
        super().fplot_mult(properties)


    def fplot_ravg(self, properties):
        super().fplot_ravg(properties)




class Simulate3D(BaseFunctions, Reader):
    def __init__(self, deltaT=None, density=None, temperature=None, rCut=None, initUcellx=None, initUcelly=None, initUcellz=None,
                nebrTabFac=None, rNebrShell=None, stepAvg=None, stepEquil=None, stepLimit=None, stepAdjustTemp=None, limitVel=None,
                rangeVel=None, sizeHistVel=100, stepVel=None, randSeedP=None, method='Cell-Neighbor', printout=False, outfile='output.txt'):
        if printout == True or printout == 1:
            pout = 1
        else:
            pout = 0
        methods = {'All-pair':      0,
                   'Cell-list':     1,
                   'Neighbor-list': 2,
                   'Cell-Neighbor': 3}
        method_ = methods[method]
        if not outfile.endswith(tuple(['.txt'])):
            outfile += '.txt'
        with open('input.txt', 'w') as in_file:
            in_file.write(str(deltaT) + '-----DeltaT\n')
            in_file.write(str(density) + '-----density\n')
            in_file.write(str(temperature) + '-----temperature\n')
            in_file.write(str(rCut) + '-----rCut\n')
            in_file.write(str(initUcellx) + '-----initUcellx\n')
            in_file.write(str(initUcelly) + '-----initUcelly\n')
            in_file.write(str(initUcellz) + '-----initUcellz\n')
            in_file.write(str(nebrTabFac) + '-----nebrTabFac\n')
            in_file.write(str(rNebrShell) + '-----rNebrShell\n')
            in_file.write(str(stepAvg) + '-----stepAvg\n')
            in_file.write(str(stepEquil) + '-----stepEquil\n')
            in_file.write(str(stepLimit) + '-----stepLimit\n')
            in_file.write(str(stepAdjustTemp) + '-----stepAdjustTemp\n')
            in_file.write(str(limitVel) + '-----limitVel\n')
            in_file.write(str(rangeVel) + '-----rangeVel\n')
            in_file.write(str(sizeHistVel) + '-----sizeHistVel\n')
            in_file.write(str(stepVel) + '-----stepVel\n')
            in_file.write(str(randSeedP) + '-----randSeedP\n')
            in_file.write(str(method_) + '-----method\n')
            in_file.write(str(pout) + '-----printout\n')
            in_file.write(str(outfile) + '\n')
        MD3D.MD3D('input.txt')
        self.outputs = {}
        self.outputs[method] = Reader(outfile=outfile)


    def plot_prob_vs_vel(self, increments=[0, 1, 2, 5, 10]):
        super().plot_prob_vs_vel(increments)


    def plot_hfun_vs_time(self, low_r=0, up_r=10):
        super().plot_hfun_vs_time(low_r, up_r)


    def fplot(self, property, Lmean=True):
        super().fplot(property, Lmean)


    def fplot_mult(self, properties):
        super().fplot_mult(properties)


    def fplot_ravg(self, properties):
        super().fplot_ravg(properties)




class ReadfromOutput(BaseFunctions):
    def __init__(self, outfiles):
        self.outputs = {}
        for output in outfiles:
            self.outputs[output] = Reader(output)


    def plot_prob_vs_vel(self, increments=[0, 1, 2, 5, 10], output_list=None):
        super().plot_prob_vs_vel(increments, output_list)


    def plot_hfun_vs_time(self, low_r=0, up_r=10, output_list=None):
        super().plot_hfun_vs_time(low_r, up_r, output_list)


    def fplot(self, property, Lmean=True, output_list=None):
        super().fplot(property, Lmean, output_list)


    def fplot_mult(self, properties, output_list=None):
        super().fplot_mult(properties, output_list)


    def fplot_ravg(self, properties, output_list=None):
        super().fplot_ravg(properties, output_list)



def RunCases2D(infiles=['input.txt']):
    for infile in infiles:
        if not infile.endswith(tuple(['.txt'])):
            infile += '.txt'
        MD2D.MD2D(infile)



def RunCases3D(infiles=['input.txt']):
    for infile in infiles:
        if not infile.endswith(tuple(['.txt'])):
            infile += '.txt'
        MD3D.MD3D(infile)
