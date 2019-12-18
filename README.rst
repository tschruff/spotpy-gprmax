=============
SPOTPY-gprMax
=============


Aim
---

The aim of the project is to allow gprMax_ to be used as the simulation component in SPOTPY_.

Particularly we want to achieve maximum possible parallelism of both packages. For example if using the SCEUA optimiser
of SPOTPY we want to be able to use the functionality to allow each SCE complex to be a MPI task. With gprMax we want to
be able to exploit the MPI task farming of models with either CPU-based (OpenMP) or GPU-based (CUDA) solving. This will
most likely require some sort of two level MPI, perhaps with MPI.Spawn.


Getting the code
----------------

The first thing you need is a copy of the SPOTPY and gprMax source files. The easiest way is to get them directly from
github via ``clone``. Assuming you're in the SPOTPY-gprMax main folder (the folder where this file resides) just do the
following to download gprMax and checkout the correct branch ::

    git clone git@github.com:tschruff/gprMax.git
    cd gprMax
    git checkout jureca
    cd ..

And to download SPOTPY you just do ::

    git clone git@github.com:tschruff/spotpy.git
    cd spotpy
    git checkout custom_mpi_comm
    cd ..

Please note, that the source files are currently located in the personal (and public) repository of Tobias Schruff,
since the source files of SPOTPY and gprMax have been slightly modified in order to make the coupled SPOTPY-gprMax run
on JURECA. However, this may change in the future (as open pull requests get accepted) and source locations may have to
be adopted. Please see the section `TODO`_ for more details on open pull requests.

You should now have a ``spotpy`` and ``gprMax`` folder in the main SPOTPY-gprMax folder. If so, you're ready to proceed
with the installation of SPOTPY and gprMax. The installation procedure depends on the machine. Currently only JURECA is
supported.


Install
-------

You can install SPOTPY and gprMax on JURECA by simply running the setup script ``setup.sh`` by
typing ::

    ./setup.sh

in the terminal. The script will load all required system modules (e.g. Python), install a virtual Python environment
(into the folder ``venv``), install all required Python modules (see requirements.txt), and finally install SPOTPY and
gprMax into the virtual environment as well.

Please note that because we don't install the Python modules in the system Python interpreter (which is not possible due
to insufficient rights) you must activate the virtual environment before you can use SPOTPY, gprMax, or any other Python
module that we just installed via ::

    source env2019a.sh
    source venv/bin/activate

You can simply deactivate the virtual environment by typing ``deactivate`` in the terminal window. You must also make sure
to load the system modules via ``source env2019a.sh`` before you can use Python and before you active the virtual environment.


Run
---

    NOTE: GPU mode is not yet implemented!

You can run SPOTPY-gprMax by using the supplied batch scripts ``spotpy-gprmax-cpu.sh`` and ``spotpy-gprmax-gpu.sh`` via ::

    sbatch spotpy-gprmax-cpu.sh

Please check out the respective batch scripts and adjust the settings to your needs before you submit a job.


MPI code sections
-----------------

The MPI Spawn approach in gprMax exists in the ``gprMax.py`` module with the function ``run_mpi_sim`` around line 345

The MPI code for SPOTPY exists in the ``parallel/mpi.py`` part of the package.

TODO
----

- SPOTPY pull request #242 still pending (https://github.com/thouska/spotpy/pull/242).
  Adjust SPOTPY source in `Getting the code`_ once it is accepted.
- gprMax pull request #233 still pending (https://github.com/gprMax/gprMax/pull/233).
  Adjust gprMax source in `Getting the code`_ once it is accepted.
- GPU mode not implemented in ``run.py``


Documentation
-------------

.. _gprMax: http://docs.gprmax.com
.. _SPOTPY: http://fb09-pasig.umwelt.uni-giessen.de/spotpy/
