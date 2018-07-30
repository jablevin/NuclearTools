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

U = pint.UnitRegistry()

#constants
NA = 6.0221409 * 10**23
nuclide_data = get_python_lib() + '/NuclearTools/Nuclide_Data.txt'
Pm = 1.0072764669
Nm = 1.00866491588
em = .0005485799
MT_dict = {'(n,2nd)'     :11,
        '(n,2n)'         :16,
        '(n,3n)'         :17,
        '(n,nalpha)'     :22,
        '(n,2nalpha)'    :24,
        '(n,np)'         :28,
        '(n,n2alpha)'    :29,
        '(n,nd)'         :32,
        '(n,nt)'         :33,
        '(n,2np)'        :41,
        '(n,3np)'        :42,
        '(n,n2p)'        :44,
        '(n,npalpha)'    :45,
        '(n,nprime)'     :64,
        '(n,nprime)'     :91,
        '(n,p)'          :3,
        '(n,d)'          :4,
        '(n,t)'          :5,
        '(n,alpha)'      :7,
        '(n,2alpha)'     :8,
        '(n,2p)'         :11,
        '(n,palpha)'     :12,
        '(n,pd)'         :15,
        '(n,pt)'         :16,
        '(n,dalpha)'     :17,
        '(n,p)'          :103,
        '(n,d)'          :50,
        '(n,d)'          :99,
        '(n,t)'          :0,
        '(n,t)'          :49,
        '(n,alpha)'      :849,
        'fission'        :18,
        '(n,f)'          :18,
        '(n,gamma)'      :102 }

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
    """ Provides atomic mass of atom.  Input as Cs """
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
def num_density(material, compound = False, density = None, weight_per = None, atom_per = None):
    """ Provides the number density of a lone atom material or a compound """
    if not compound:
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



def BE_per_nucleon(atom):
    """ Finds the binding energy of a given atom in MeV"""
    real_mass = atomic_mass(atom)
    index = atom.index('-')
    element = str(atom[0:index])
    A = int(atom[index+1:])
    num_Z = atomic_number(element)
    num_N = A - num_Z
    exp_mass = Pm * num_Z + Nm * num_N
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



# class weight_fraction(object):
#     """ Converts weight fraction to atom fraction"""
#     def __init__(self, weights, elements, density, compound = None):
#         assert len(weights) == len(elements), (
#                 "Number of weight fractions does not equal number of elements")
#         self.weights = []
#         self.elements = []
#         self.indv_mass = []
#         self.density = density
#
#         if compound == 'UO2':
#             self.comp_mass = Element('U').atomic_mass + 2 * Element('O').atomic_mass
#             self.elem_mass = Element('U').atomic_mass
#
#         for i in range(len(weights)):
#             self.weights.append(weights[i])
#             self.elements.append(elements[i])
#
#         for name in self.elements:
#             self.indv_mass.append(atomic_mass(name))
#
#     @property
#     def num_density(self):
#         self.number_density = []
#         for i in range(len(self.weights)):
#             self.number_density.append( self.weights[i] * NA * self.density / self.indv_mass[i]
#                     * (self.elem_mass / self.comp_mass) )
#         return self.number_density



class cross_section(object):
    """
    Finds cross sections for input nuclide and can plot or resamble to given energy groups
    """

    def __init__(self, nuclide_lookup, MT, MF):
        try:
            self.MT = int(MT)
        except:
            self.MT = int(MT_dict[MT])

        self.MF = MF

        site = "https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/"
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
        self.energies, self.cross_sections = self.read_table(sec)

    def read_float(self, v):
        """
        Convert ENDF6 string to float
        """
        if v.strip() == '':
            return 0.
        try:
            return float(v)
        except ValueError:
            # ENDF6 may omit the e for exponent
            return float(v[0] + v[1:].replace('+', 'e+').replace('-', 'e-'))  # don't replace leading negative sign


    def read_line(self, l):
        """Read first 6*11 characters of a line as floats"""
        return [self.read_float(l[s]) for s in self.slices['data']]


    def read_table(self, lines):
        """ Parse Data """
        f = self.read_line(lines[1])
        nS = int(f[4])  # number of interpolation sections
        nP = int(f[5])  # number of data points

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
        cmpstr = '%2s' % MF       # search string
        i0 = v.index(cmpstr)            # first occurrence
        i1 = n - v[::-1].index(cmpstr)  # last occurrence
        return lines[i0: i1]


    def find_section(self, lines, MF, MT):
        """Locate and return a certain section"""
        v = [l[70:75] for l in lines]
        n = len(v)
        cmpstr = '%2s%3s' % (MF, MT)       # search string
        i0 = v.index(cmpstr)            # first occurrence
        i1 = n - v[::-1].index(cmpstr)  # last occurrence
        return lines[i0: i1]


    def list_content(self, lines):
        """Return set of unique tuples (MAT, MF, MT)"""
        s0 = self.slices['MAT']
        s1 = self.slices['MF']
        s2 = self.slices['MT']
        content = set(((int(l[s0]), int(l[s1]), int(l[s2])) for l in lines))

        # remove section delimiters
        for c in content.copy():
            if 0 in c:
                content.discard(c)
        return content

    def plot(self):
        plt.figure(figsize=(10,6))
        plt.loglog(self.energies[1:], self.cross_sections[1:], color = 'darkorange')
        plt.xlabel('Energies [eV]')
        plt.ylabel('Cross Section [barns]')
        plt.tight_layout()
        plt.show()
        return None

    def single_average(self, func, func_units):
        if func_units == 'MeV':
            energy = self.energies / 10**6
        if func_units == 'keV':
            energy = self.energies / 10**3

        function = []
        for i in range(len(energy)):
            function.append(func(energy[i]))

        inner = []
        for i in range(len(energy)):
            inner.append(self.cross_sections[i]*function[i])

        numerator = np.trapz(inner, energy)
        denominator = np.trapz(function, energy)
        return numerator / denominator

    # TODO Add ability to condense down to energy group Ex: fast and thermal groups.
