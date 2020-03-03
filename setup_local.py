from distutils.sysconfig import get_python_lib
import os

pythonlib = get_python_lib()
pythonlib = pythonlib.replace('\\', '/')
cwd = os.getcwd()
cwd = cwd.replace('\\', '/')

newpath = pythonlib
if not os.path.exists(newpath):
    os.makedirs(newpath)

# try:
#     # print('cp \'' + cwd + '/NuclearTools/Tools.py\' ' + newpath)
#     os.system('cp \'' + cwd + '/NuclearTools/Tools.py\' ' + newpath)
#     os.system('cp \'' + cwd + '/NuclearTools/Nuclide_Data.txt\' ' + newpath)
#     os.system('cp \'' + cwd + '/NuclearTools/__init__.py\' ' + newpath)
#     os.system('cp \'' + cwd + '/NuclearTools/ThermalHydraulics.py\' ' + newpath)
#     os.system('cp \'' + cwd + '/NuclearTools/MassFlux.py\' ' + newpath)
#     os.system('cp \'' + cwd + '/NuclearTools/Natural_Circulation_SMR.py\' ' + newpath)
# except:
#     pass

try:
    os.system('cp -r \'' + cwd + '/build/lib.win-amd64-3.7/NuclearTools\' ' + newpath + '/')
    # print('cp -r \'' + cwd + '/build/lib.win-amd64-3.7/NuclearTools\' ' + newpath + '/')
except:
    pass
