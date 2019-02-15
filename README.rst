***
Aim
***

The aim of the project is to allow gprMax to be used as the simulation component in SPOTPY.

Particularly we want to achieve maximum possible parallelism of both packages. For example if using the SCEUA optimiser of SPOTPY we want to be able to use the functionality to allow each SCE complex to be a MPI task. With gprMax we want to be able to exploit the MPI task farming of models with either CPU-based (OpenMP) or GPU-based (CUDA) solving. This will most likely require some sort of two level MPI, perhaps with MPI.Spawn.

*************
Getting setup
*************

Documentation for gprMax: http://docs.gprmax.com

Documentation for SPOTPY: http://fb09-pasig.umwelt.uni-giessen.de/spotpy/

1. If testing on Jureca ensure the following modules are loaded **before** creating the conda environment and compiling gprMax. This ensures packages such as ``mpi4py`` and gprMax and compiled with the correct compilers loaded: ``module load intel-para/2017b.1-mt CUDA/9.0.176``
2. Install Miniconda: http://conda.pydata.org/miniconda.html
3. Create a conda environment: ``conda env update -f conda_env.yml``
4. Clone gprMax repository: ``git clone https://github.com/gprMax/gprMax.git``
5. Build and install gprMax: ``cd gprMax; python setup.py build; python setup.py install``

A batch script for submitting to Jureca is ``spotpy-gprmax.sh`` and should be edited depending on the parallel setup being tested.

The Python module that runs SPOTPY (and is called from the batch script) is ``spotpy_run_gprMax.py``. Add argument ``parallel='mpi'`` to the sceua function to enable SPOTPY to parallelise the SCE complexes, i.e. each complex is a MPI task. The ``ngs`` argument to the sample function sets the number of complexes. The number of MPI tasks specified in the ``spotpy-gprmax.sh`` batch script, should be the number of SCE complexes (workers) + 1 (master), i.e. if there are 2 SCE complexes, then the number of MPI tasks is 3.

The Python module ``cylinder_Bscan_2D_gprMax.py`` in the setups directory, configures SPOTPY and from here gprMax and associated input/output files are setup. Currently the mechanism to get a new parameter set into gprMax is to write them to file and then pass that as an input file to gprMax.

The line in ``cylinder_Bscan_2D_gprMax.py`` that calls gprMax can have arguments that enable solving on GPU and to enable the MPI functionality in gprMax. These arguments are explained in the gprMax docs: http://docs.gprmax.com/en/latest/include_readme.html#optional-command-line-arguments

*****************
MPI code sections
*****************

The MPI Spawn approach in gprMax exists in the ``gprMax.py`` module with the function ``run_mpi_sim`` around line 345

The MPI code for SPOTPY exists in the ``parallel/mpi.py`` part of the package.
