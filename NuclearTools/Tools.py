import pint
import numpy as np
import datetime
import os
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import matplotlib.pyplot as plt
from distutils.sysconfig import get_python_lib
import urllib
from scipy.interpolate import interp1d

U = pint.UnitRegistry()

#constants
NA = 6.0221409 * 10**23
nuclide_data = get_python_lib() + '/NuclearTools/Nuclide_Data.txt'
Pm = 1.0072764669
Hm = 1.00782503207
Nm = 1.00866491588
em = .0005485799
MT_dict = {'(z,total)'   :1,
        '(z,EL)'         :2,
        '(z,NEL)'        :3,
        '(z,n)'          :4,
        '(z,t)'          :5,
        '(z,continuum)'  :10,
        '(z,2nd)'        :11,
        '(z,2n)'         :16,
        '(z,3n)'         :17,
        '(z,fission)'    :18,
        '(z,f)'          :19,
        '(z,nf)'         :20,
        '(z,2nf)'        :21,
        '(z,na)'         :22,
        '(z,n3a)'        :23,
        '(z,2na)'        :24,
        '(z,3na)'        :25,
        '(z,abs)'        :27,
        '(z,np)'         :28,
        '(z,n2a)'        :29,
        '(z,2n2a)'       :30,
        '(z,nd)'         :32,
        '(z,nt)'         :33,
        '(z,nHe-3)'      :34,
        '(z,nd2a)'       :35,
        '(z,nt2a)'       :36,
        '(z,4n)'         :37,
        '(z,3nf)'        :38,
        '(z,2np)'        :41,
        '(z,3np)'        :42,
        '(z,n2p)'        :44,
        '(z,npa)'        :45,
        '(z,n0)'         :50,
        '(z,nc)'         :91,
        '(z,disap)'      :101,
        '(z,gamma)'      :102,
        '(z,p)'          :103,
        '(z,alpha)'      :849,

        'total'          :1,
        'elastic'        :2,
        'nonelastic'     :3,
        'anything'       :5,
        'fission'        :18,}


def indv_elements(compound):
    """ Returns a list of individual elements and a list of their multiplicities"""
    elements = []
    inst = []
    array = []
    j = -1
    for x in compound:
        if str(x).isupper() and type(x) != int:
            array.append(x)
            j += 1
        elif not str(x).isupper() and type(x) != int:
            array[j] = array[j] + x
    inst = [1 for i in range(len(array))]
    index = 0
    for el in array:
        j = 0
        cont = True
        while cont:
            try:
                inst[index] = (int(el[j:]))
                index += 1
                cont = False
            except:
                j += 1
                cont = True
            if j == len(el):
                cont = False
                index += 1
            if j > 15:
                cont = False
    for i in range(len(array)):
        el = array[i]
        elements.append(el.strip(str(inst[i])))
    return elements, inst



def nuclide_dict(temp = None):
    """ Returns a dictionary of all isotopes pointing to blank numpy arrays"""
    dict = {}
    with open(nuclide_data) as search:
        lines = search.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip()
        for i in range(len(lines)):
            if 'Atomic Symbol' in lines[i]:
                dict[str(lines[i][16:]) + '-' + str(lines[i+1][14:])] = np.array([])
    return dict



def atomic_mass(atom):
    """ Provides atomic mass of atom.  Input as Cs-137 """
    index = atom.index('-')
    element = str(atom[0:index])
    A = str(atom[index+1:])
    with open(nuclide_data) as search:
        lines = search.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip()
        for i in range(len(lines)):
            if lines[i] == 'Atomic Symbol = ' + element and lines[i+1] == 'Mass Number = ' + A:
                break
    return float(lines[i+2][23:(lines[i+2].index('('))])



def standard_mass(element):
    """ Provides standard mass of atom.  Input as Cs """
    with open(nuclide_data) as search:
        lines = search.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip()
        for i in range(len(lines)):
            if lines[i] == 'Atomic Symbol = ' + element:
                break
    try:
        return float(lines[i+4][25:(lines[i+4].index('('))])
    except:
        try:
            return float(lines[i+4][26:(lines[i+4].index(']'))])
        except:
            raise Exception('Standard mass not available for ' + element)



def atomic_number(element):
    """ Provides atomic number of atom.  Input as Cs """
    with open(nuclide_data) as search:
        lines = search.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip()
        for i in range(len(lines)):
            if lines[i] == 'Atomic Symbol = ' + element:
                break
    return int(lines[i-1][16:])



def molec_mass(compound):
    """ Input molecular compound such as UO2 for Uranium Oxide to obtain molecular mass """
    mass = []
    elements, inst = indv_elements(compound)
    for i in range(len(elements)):
        try:
            mass.append(standard_mass(elements[i]) * inst[i])
        except:
            raise Exception("Element " + str(elements[i]) + " is not a known element")

    return sum(mass)



# TODO finish material number densities
def num_density(material, is_compound = False, density = None, weight_per = None, atom_per = None):
    """ Provides the number density of a lone atom material or a compound """
    if not is_compound:
        index = material.index('-')
        element = str(material[0:index])
        A = int(material[index+1:])
        rho = density
    else:
        if weight_per != None:
            print('h')

        elif atom_per != None:
            elements, inst = indv_elements(material)

        else:
            raise Exception("Weight nor atom percent given for compound")

    return (rho * NA / A) / U.cm**3


# class material_fractions(object):
#     """ Converts weight fraction to atom fraction"""
#     def __init__(self, material, is_compound=False, density=None, weight_percent=None, atom_percent=None, isotopes=None):
#         self.weight_percent = weight_percent
#         self.atom_percent = atom_percent
#         self.density = density
#         self.elements, self.inst = indv_elements(material)
#         self.indv_mass = [atomic_mass(i) for i in self.elements]
#         holder = [standard_mass(i) for i in self.elements]
#         self.comp_mass = sum([holder * self.inst[i] for i in range(len(self.inst))])
#         self.isotopes = isotopes
#
#         if element_of_interest == None:
#             element_of_interest = self.elements[0]
#
#         self.elem_mass = standard_mass(element_of_interest)
#
#         # if compound == 'UO2':
#         #     self.comp_mass = Element('U').atomic_mass + 2 * Element('O').atomic_mass
#         #     self.elem_mass = Element('U').atomic_mass
#
#         # for name in self.elements:
#         #     self.indv_mass.append(atomic_mass(name))
#
#     @property
#     def compound_number_density(self):
#         self.number_density = []
#         for i in range(len(self.weight_percent)):
#             self.number_density.append( self.weight_percent[i] * NA * self.density / self.indv_mass[i]
#                     * (self.elem_mass / self.comp_mass) )
#         return self.number_density


def BE_per_nucleon(atom):
    """ Finds the binding energy of a given atom in MeV"""
    real_mass = atomic_mass(atom)
    index = atom.index('-')
    element = str(atom[0:index])
    A = int(atom[index+1:])
    num_Z = atomic_number(element)
    num_N = A - num_Z
    exp_mass = Hm * num_Z + Nm * num_N
    mass_defect = exp_mass - real_mass
    return (931.5 * (mass_defect)) / A * U.MeV



def Q_value(reactants, products):
    """ Provides the Q value in Mev from a list of reactants and a list of products """
    temp1, temp2 = 0, 0
    for reactant in reactants:
        if reactant[-1:] == 'n':
            try:
                temp1 += int(reactant[0:-1]) * Nm
            except:
                temp1 += Nm
        elif reactant[-1:] == 'p':
            try:
                temp1 += int(reactant[0:-1]) * Pm
            except:
                temp1 += Pm
        elif reactant[-1:] == 'e':
            try:
                temp1 += int(reactant[0:-1]) * em
            except:
                temp1 += em
        else:
            temp1 += atomic_mass(reactant)

    for product in products:
        if product[-1:] == 'n':
            try:
                temp2 += int(product[0:-1]) * Nm
            except:
                temp2 += Nm
        elif product[-1:] == 'p':
            try:
                temp2 += int(product[0:-1]) * Pm
            except:
                temp2 += Pm
        elif product[-1:] == 'e':
            try:
                temp2 += int(product[0:-1]) * em
            except:
                temp2 += em
        else:
            temp2 += atomic_mass(product)

    return (temp1 - temp2) * 931.5 * U.MeV



def coh_scatter_energy(atom, angle, E):
    """ Provides the final energy after coherent (elastic) scattering """
    index = atom.index('-')
    element = str(atom[0:index])
    A = int(atom[index+1:])
    E_prime = (1 / (A + 1)**2) * ( np.sqrt(E) * np.cos(np.radians(angle)) +
            np.sqrt(E * (A**2 - 1 + (np.cos(np.radians(angle)))**2)))**2
    return E_prime




class cross_section(object):
    """
    Finds cross sections for input nuclide and can plot or resamble to given energy groups
    """

    def __init__(self, nuclide_lookup, MF, MT):
        try:
            index_MT0 = MT.index('(')
            index_MT1 = MT.index(',')
            incident = MT[1+index_MT0:index_MT1]
        except:
            incident = 'n'

        try:
            self.MT = int(MT)
        except:
            try:
                self.MT = int(MT_dict[MT])
            except:
                tmp_string = '(z' + MT[index_MT1:]
                self.MT = int(MT_dict[tmp_string])

        self.MF = MF

        site = "https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/" + incident + "/"
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers={'User-Agent':user_agent,}

        request = urllib.request.Request(site,None,headers)
        response = urllib.request.urlopen(request)
        filestream = response.read().decode('utf-8')

        index0 = filestream.index(nuclide_lookup)
        index1 = filestream[index0::-1].index('_n')
        index2 = filestream[index0:].index('zip')
        split = slice(index0-index1-1, index2+index0+3)

        request = urllib.request.Request(site + filestream[split], None, headers)
        response = urllib.request.urlopen(request)
        zipfile = ZipFile(BytesIO(response.read()))
        with open((get_python_lib() + '/NuclearTools/temp.dat'), 'w') as file:
            for line in zipfile.open(zipfile.namelist()[0]).readlines():
                file.write(line.decode('utf-8').rstrip() + '\n')

        del zipfile, request, response

        self.slices = {
            'MAT': slice(66, 70),
            'MF': slice(70, 72),
            'MT': slice(72, 75),
            'line': slice(75, 80),
            'content': slice(0, 66),
            'data': (slice(0, 11), slice(11, 22), slice(22, 33), slice(33, 44), slice(44, 55), slice(55, 66))}

        f = open(get_python_lib() + '/NuclearTools/temp.dat')
        lines = f.readlines()
        sec = self.find_section(lines, self.MF, self.MT)
        self.indep_var, self.cross_sections = self.read_table(sec)

    def read_float(self, v):
        """
        Convert ENDF6 string to float
        """
        if v.strip() == '':
            return 0.
        try:
            return float(v)
        except ValueError:
            return float(v[0] + v[1:].replace('+', 'e+').replace('-', 'e-'))


    def read_line(self, l):
        """Read first 6*11 characters of a line as floats"""
        return [self.read_float(l[s]) for s in self.slices['data']]


    def read_table(self, lines):
        """ Parse Data """
        f = self.read_line(lines[1])
        nS = int(f[4])
        nP = int(f[5])

        # data lines
        x = []
        y = []
        for l in lines[3:]:
            f = self.read_line(l)
            x.append(f[0])
            y.append(f[1])
            x.append(f[2])
            y.append(f[3])
            x.append(f[4])
            y.append(f[5])
        return np.array(x[:nP]), np.array(y[:nP])


    def find_file(self, lines, MF):
        """Locate and return a certain file"""
        v = [l[self.slices['MF']] for l in lines]
        n = len(v)
        cmpstr = '%2s' % MF
        i0 = v.index(cmpstr)
        i1 = n - v[::-1].index(cmpstr)
        return lines[i0: i1]


    def find_section(self, lines, MF, MT):
        """Locate and return a certain section"""
        v = [l[70:75] for l in lines]
        n = len(v)
        cmpstr = '%2s%3s' % (MF, MT)
        i0 = v.index(cmpstr)
        i1 = n - v[::-1].index(cmpstr)
        return lines[i0: i1]


    def list_content(self, lines):
        """Return set of unique tuples (MAT, MF, MT)"""
        s0 = self.slices['MAT']
        s1 = self.slices['MF']
        s2 = self.slices['MT']
        content = set(((int(l[s0]), int(l[s1]), int(l[s2])) for l in lines))

        for c in content.copy():
            if 0 in c:
                content.discard(c)
        return content


    def plot(self):
        plt.figure(figsize=(10,6))
        plt.loglog(self.indep_var[1:]/10**6, self.cross_sections[1:], color = 'darkorange')
        if self.MF == 3:
            plt.xlabel('Energies [MeV]')
        plt.ylabel('Cross Section [barns]')
        plt.tight_layout()
        plt.show()

    def single_average(self, func, func_units):
        if func_units == 'MeV':
            indep_var = self.indep_var / 10**6
        if func_units == 'keV':
            indep_var = self.indep_var / 10**3
        if func_units == 'eV':
            indep_var = self.indep_var

        function = []
        for i in range(len(indep_var)):
            function.append(func(indep_var[i]))

        inner = []
        for i in range(len(indep_var)):
            inner.append(self.cross_sections[i]*function[i])

        numerator = np.trapz(inner, indep_var)
        denominator = np.trapz(function, indep_var)
        return numerator / denominator

    def make_csv(self, filename=None):
        tmp = np.vstack((self.indep_var, self.cross_sections))
        txt = tmp.transpose()
        location = os.getcwd()
        if filename == None:
            filename = 'cross_section_data.csv'

        if filename[:-4] != '.csv':
            filename += '.csv'

        np.savetxt(location + '/' + filename, txt, delimiter=',')


    # TODO Add ability to condense down to several energy groups Ex: fast and thermal groups.
    # def condense_to(self, groups):
    #     groups = int(groups)
    #     group_len = len(self.cross_sections) // groups


    def cross_section_mev(self, energy, method='linear'):
        energy = energy * 10**6
        absolute_difference_function = lambda list_value : abs(list_value - energy)
        closest_value = min(self.indep_var, key=absolute_difference_function)
        index = self.indep_var.tolist().index(closest_value)
        inter = interp1d(self.indep_var[index-1:index+1], self.cross_sections[index-1:index+1], kind=method)
        return inter(energy)



class Stopping_Power(object):
    """ Calculates the stopping power of provided particles in a given medium and plots over a specified range """
    def __init__(self, medium, particles, rho, U, I):
        # knowns
        self.U = U
        self.r = 2.818 * 10**(-15) * self.U.m
        self.c = 299792458 * self.U.m/self.U.s
        self.me = 9.10938356*10**(-31) * self.U.kg
        self.mec2 = 0.510998950 * self.U.MeV
        self.NA = 6.0221409 * 10**23 * self.U.mole**(-1)

        # inputs
        self.medium = medium
        self.particles = particles
        self.rho = rho
        self.elements, self.inst = indv_elements(self.medium)

        self.A_medium, self.Z_medium = {}, {}
        self.A_total = 0 * self.U.gram / self.U.mole
        index = 0
        for i in self.elements:
            self.A_medium[i] = standard_mass(i) * self.U.gram / self.U.mole
            self.A_total += standard_mass(i) * self.U.gram / self.U.mole * self.inst[index]
            self.Z_medium[i] = atomic_number(i)
            index += 1

        self.Z_particles, self.M_particles = {}, {}
        for i in self.particles:
            self.M_particles[i] = atomic_mass(i) * self.U.amu
            try:
                self.Z_particles[i] = atomic_number(indv_elements(i)[0][0])
            except:
                pass

        self.I = {}
        i = 0
        for m in self.elements:
            self.I[m] = I[i]
            i += 1

        self.energies = {}
        for i in self.particles:
            if i in ['ep-0', 'e-0']:
                self.energies[i] = np.logspace(-2, 4, num=100)
            else:
                self.energies[i] = np.logspace(0, 4, num=100)

    def L(self, m, T):
        """ lorentz Factor """
        return T / (m * self.c**2) + 1

    def v(self, m, T):
        """ Finding velocity from mass and energy of particle """
        return self.c * np.sqrt(1 - (1 / (T / (m * self.c**2) + 1 ) )**2 )

    def SP_heavy(self, z, Z, A, I, T, m):
        """ Stopping power for proton, deuterons, tritons and alpha """
        return (4 * np.pi * self.r**2 * z**2 * (self.mec2 / (self.v(m, T) / self.c)**2) * self.NA / A * Z *
               (np.log(2 * (self.mec2 / I) * (self.v(m, T) / self.c)**2 * self.L(m, T)**2) -
               (self.v(m, T) / self.c)**2))

    def SP_electron(self, Z, A, I, T, m):
        """ Stopping power for electrons """
        return (4 * np.pi * self.r**2 * (self.mec2 / (self.v(m, T) / self.c)**2) * self.NA / A * Z *
                (np.log( self.mec2 / I * self.v(m, T) / self.c * self.L(m, T) * np.sqrt(self.L(m, T) - 1)) +
                1 / (2 * self.L(m, T)**2) * ( (self.L(m, T) - 1)**2 / 8 + 1 -
                (self.L(m, T)**2 + 2 * self.L(m, T) - 1) * np.log(2) ) ) )

    def SP_positron(self, Z, A, I, T, m):
        """ Stopping power for positron """
        return (4 * np.pi * self.r**2 * (self.mec2 / (self.v(m, T) / self.c)**2) * self.NA / A * Z *
                (np.log( self.mec2 / I * self.v(m, T) / self.c * self.L(m, T) * np.sqrt(self.L(m, T) - 1)) -
                (self.v(m, T) / self.c)**2 / 24 * (23 + 14 / (self.L(m, T) + 1) + 10 / (self.L(m, T) + 1)**2 +
                4 / (self.L(m, T) + 1)**3) + np.log(2) / 2 ) )


    def SPatE(self, E):
        """ Returns the stopping power for each particle at given energy (E) """
        self.SP = {}
        keylist = list(self.A_medium.keys())
        index = 0
        for particle in self.particles:
            if particle in ['ep-0']:
                self.temp = []
                for m in self.elements:
                    self.temp.append(self.SP_positron(self.Z_medium[m], self.A_medium[m], self.I[m], E, self.M_particles[particle]))
                temp2 = 0
                for j in range(len(self.A_medium)):
                    temp2 += self.A_medium[keylist[j]] * self.inst[j] / self.A_total * self.temp[j]
                self.SP[particle] = ((temp2 * self.rho).to(self.U.MeV/self.U.cm))
            elif particle in ['e-0']:
                self.temp = []
                for m in self.elements:
                    self.temp.append(self.SP_electron(self.Z_medium[m], self.A_medium[m], self.I[m], E, self.M_particles[particle]))
                temp2 = 0
                for j in range(len(self.A_medium)):
                    temp2 += self.A_medium[keylist[j]] * self.inst[j] / self.A_total * self.temp[j]
                self.SP[particle] = ((temp2 * self.rho).to(self.U.MeV/self.U.cm))
                index += 1
            else:
                self.temp = []
                for m in self.elements:
                    self.temp.append(self.SP_heavy(self.Z_particles[particle], self.Z_medium[m], self.A_medium[m], self.I[m], E, self.M_particles[particle]))
                temp2 = 0
                for j in range(len(self.A_medium)):
                    temp2 += self.A_medium[keylist[j]] * self.inst[j] / self.A_total * self.temp[j]
                self.SP[particle] = ((temp2 * self.rho).to(self.U.MeV/self.U.cm))
                index += 1
        return self.SP


    def plot(self, xrange, yrange):
        """ Plots the stopping power vs. energy over specified range """
        self.SP = {}
        keylist = list(self.A_medium.keys())
        index = 0
        for particle in self.particles:
            self.SP[particle] = []
            if particle in ['ep-0']:
                for i in self.energies[particle]:
                    self.temp = []
                    for m in self.elements:
                        self.temp.append(self.SP_positron(self.Z_medium[m], self.A_medium[m], self.I[m], i * self.U.MeV, self.M_particles[particle]))
                    temp2 = 0
                    for j in range(len(self.A_medium)):
                        temp2 += self.A_medium[keylist[j]] * self.inst[j] / self.A_total * self.temp[j]
                    self.SP[particle].append((temp2 * self.rho).to(self.U.MeV/self.U.cm).magnitude)
                    index += 1
            elif particle in ['e-0']:
                for i in self.energies[particle]:
                    self.temp = []
                    for m in self.elements:
                        self.temp.append(self.SP_electron(self.Z_medium[m], self.A_medium[m], self.I[m], i * self.U.MeV, self.M_particles[particle]))
                    temp2 = 0
                    for j in range(len(self.A_medium)):
                        temp2 += self.A_medium[keylist[j]] * self.inst[j] / self.A_total * self.temp[j]
                    self.SP[particle].append((temp2 * self.rho).to(self.U.MeV/self.U.cm).magnitude)
                    index += 1
            else:
                for i in self.energies[particle]:
                    self.temp = []
                    for m in self.elements:
                        self.temp.append(self.SP_heavy(self.Z_particles[particle], self.Z_medium[m], self.A_medium[m], self.I[m], i * self.U.MeV, self.M_particles[particle]))
                    temp2 = 0
                    for j in range(len(self.A_medium)):
                        temp2 += self.A_medium[keylist[j]] * self.inst[j] / self.A_total * self.temp[j]
                    self.SP[particle].append((temp2 * self.rho).to(self.U.MeV/self.U.cm).magnitude)
                    index += 1
        labels = {}
        for i in self.particles:
            if i in ['e-0']:
                labels[i] = '$e^-$'
            elif i in ['ep-0']:
                labels[i] = '$e^+$'
            else:
                labels[i] = '$^' + str(int(atomic_mass(i))) + '_' + str(self.Z_particles[i]) + ' ' + indv_elements(i)[0][0] + '$'
        plt.figure(figsize=(12,7))
        for i in self.particles:
            plt.plot(self.energies[i], self.SP[i], label=labels[i])
        plt.xscale('log')
        plt.xlim(xrange)
        plt.ylim(yrange)
        plt.grid(color='gray', alpha=0.6)
        plt.xlabel('kinetic energy (MeV)')
        plt.ylabel('stopping power (MeV/cm)')
        plt.legend()
        plt.tight_layout()
        plt.show()
