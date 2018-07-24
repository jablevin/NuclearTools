import pint
import numpy as np
from becquerel.tools.isotope import Isotope
from becquerel.tools.isotope_qty import IsotopeQuantity, NeutronIrradiation
from becquerel.tools.element import Element
from becquerel.tools.element import element_z
import becquerel.tools.materials as mat
import datetime
import os
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import matplotlib.pyplot as plt
from distutils.sysconfig import get_python_lib
import urllib
import Nuclear_Tools.Nuclear_Tools as nt

U = pint.UnitRegistry()

nt.atomic_mass("U-235")

nt.molec_mass('UO2')

nt.molec_mass('Y3Al6BO3Si6O18OH4')

nt.num_density('U-235')

nt.BE_per_nucleon('U-235')

nt.Q_value(['n', 'U-235'], ['Kr-92', 'Ba-141', '3n'])

nt.coh_scatter_energy('Cs-137', 25, 5 * U.MeV)

object = nt.cross_section('U-235', MT='(n,f)', MF=3)
object.cross_sections * 30
object.energies
object.plot()
