# NuclearTools
An assortment of handy nuclear engineering tools

This package provides various simple Nuclear functions to quickly obtain data in a python environment such
as giving molecular weight of compound such as UO2, giving atomic weights and binding energy per nucleon.
It can also read in specific sets of cross sections from the online ENDF library.

Other than helpful tools, the package comes equipped with various simulation methods.  The package can
simulate various heating characteristics in a BWR or PWR fuel channel.  This code is written in native
Python and therefore speed is not the goal but rather should serve as a learning experience as the
simulation will take in parameters with units and perform automatic unit conversions for ease of use.

A simulation built for speed is the molecular dynamics module which consists of both 2D and 3D
capability.  This module has the backend written in native C for speed and is linked to a python class
such that importing and changing parameters and even plotting can be easily performed.

The first goal of this package is to serve as a provider of quick assistance tools to speed up ones workflow.
The second goal is simply to serve as a learning experience for those interested in reactor thermal hydraulics
or even molecular dynamics simulation.

This package is currently in the first stages of development but plans for a quick expansion to include many
other helpful tools.

It should be noted that currently the Molecular_Dynamics module, if installed through pip,
will only work on Windows machines as a .pyd file is in this build.  An update to add
the capability for Linux and Mac OS is in the works but for the time being the distribution here can
be directly pulled and setup through the setup.py file.  setup_local.py is a file to copy the build directory
to the local python packages directory.  This shortcut may or may not work.

Please feel free to send in functions or classes that would be useful to class work or research to have
them added to the package.

# Setup

It is highly recommended to install Anaconda for the best results in compatibility and in setup.

>> pip install NuclearTools

Then you should be good to go!
