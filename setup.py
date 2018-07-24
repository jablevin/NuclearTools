from distutils.core import setup
from distutils.sysconfig import get_python_lib
import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command

pythonlib = get_python_lib()
pythonlib = pythonlib.replace('\\', '/')
cwd = os.getcwd()
cwd = cwd.replace('\\', '/')

newpath = pythonlib + '/NuclearTools'
if not os.path.exists(newpath):
    os.makedirs(newpath)

try:
    os.system('cp \'' + cwd + '/core/NuclearTools.py\' ' + newpath)
    os.system('cp \'' + cwd + '/Nuclide_Data.txt\' ' + newpath)
except:
    pass

NAME = 'NuclearTools'
DESCRIPTION = 'Handy nuclear tools for quick reference'
URL = 'https://github.com/jablevin/Nuclear_Tools'
AUTHOR = 'Jacob Blevins'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1dev'


REQUIRED = ['pint',

]

here = cwd

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
about['__version__'] = VERSION

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),

    install_requires=REQUIRED,

    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],

    cmdclass={
        'upload': UploadCommand,
    },
)
