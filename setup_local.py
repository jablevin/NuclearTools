from distutils.sysconfig import get_python_lib
import os

pythonlib = get_python_lib()
pythonlib = pythonlib.replace('\\', '/')
cwd = os.getcwd()
cwd = cwd.replace('\\', '/')

newpath = pythonlib + '/NuclearTools'
if not os.path.exists(newpath):
    os.makedirs(newpath)

try:
    os.system('cp \'' + cwd + '/NuclearTools/NuclearTools.py\' ' + newpath)
    os.system('cp \'' + cwd + '/Nuclide_Data.txt\' ' + newpath)
except:
    pass
