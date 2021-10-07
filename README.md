# MLFermiLATDwarfs

Framework to derive constraints on the (velocity-independent) dark matter pair-annihilation cross-section utilising Fermi-LAT gamma-ray data of Milky Way dwarf spheroidal galaxies (indirect detection), which features a machine learning-based assessment of astrophysical background emission (intrinsic + extrinsic) from these objects.

It is a <tt>python</tt> code to derive data-driven upper limits on the thermally averaged, velocity-weighted pair-annihilation cross-section (velocity-independent; $s$-wave) of a user-defined particle dark matter model using the expected differential gamma-ray spectrum of pair-annihilation events (provided by the user) as well as 10 years of Fermi-LAT data from observations of the Milky Way´s dwarf spheroidal galaxies.

For the documentation and a tutorial, see the provided jupyter notebook: README_analysis_rundown.ipynb

## Prerequisites

Python 3.6 or higher and the following packages:

    - numpy 
    - scipy
    - astropy
    - scikit-learn
    - iminuit (version < 2.0)

## Installation

This project can be installed as follows:

```sh
  $ git clone https://gitlab.in2p3.fr/christopher.eckner/mlfermilatdwarfs.git
  $ cd mlfermilatdwarfs
  $ pip install .
``` 

Note that the code is designed to be run via the command line interface as it requires parser arguments. However, each routine of the project maybe run on its own after the installation.
    
## License

This project is licensed under a MIT License - see the ``LICENSE`` file.

## Contact

Email to: calore [at] lapth.cnrs.fr / serpico [at] lapth.cnrs.fr / eckner [at] lapth.cnrs.fr
