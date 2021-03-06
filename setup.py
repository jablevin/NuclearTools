# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command
from distutils.core import Extension

cwd = os.getcwd()
cwd = cwd.replace('\\', '/')

NAME = 'NuclearTools'
DESCRIPTION = 'Handy nuclear tools for quick calculation and reference'
URL = 'https://github.com/jablevin/NuclearTools'
AUTHOR = 'Jacob Blevins'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.607'


REQUIRED = ['pint',
            'numpy',
            'datetime',
            'matplotlib',
            'iapws',
            'scipy',
]


try:
    with io.open(os.path.join(cwd, 'README.md'), encoding='utf-8') as f:
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
            rmtree(os.path.join(cwd, 'dist'))
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


<<<<<<< HEAD
module1 = Extension('NuclearTools.MD2D',
                    sources = ['NuclearTools/MD2D.c'])

module2 = Extension('NuclearTools.MD3D',
                    sources = ['NuclearTools/MD3D.c'])
=======
module1 = Extension('MD',
                    sources = ['MD.c'])
>>>>>>> c00b2713afa9362296e2c87a8e260c48fc6e5ec5

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
<<<<<<< HEAD
    ext_modules = [module1, module2],
=======
    ext_modules = [module1],
>>>>>>> c00b2713afa9362296e2c87a8e260c48fc6e5ec5

    install_requires=REQUIRED,
    package_data={'NuclearTools': ['Nuclide_Data.txt']},

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
