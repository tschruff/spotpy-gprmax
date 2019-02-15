import os
import sys

from mpi4py import MPI
import numpy as np
import spotpy

from gprMax.gprMax import api as gprMax
from tools.outputfiles_merge import get_output_data
from tools.outputfiles_merge import merge_files


class spotpy_setup(object):
    def __init__(self):

        # MPI Configuration for SCEUA Optimiser
        # Create new MPI communicators for each complexes
        # Split communicator into group of sub-communicators
        # (original communicator does not go away). Color determines which new
        # communicator each process will belong. All processes which pass in
        # the same value for color are assigned to the same communicator.
        # Key determines the ordering (rank) within each new communicator.
        mpicomm = MPI.COMM_WORLD
        self.worldrank = mpicomm.Get_rank()
        mpicomm.Set_name('MPILevel1-WorldRank:' + str(self.worldrank))
        print('SPOTPY/MPI ({}) - world size: {}, world rank: {}\n'.format(mpicomm.name, mpicomm.Get_size(), self.worldrank))
        color = key = self.worldrank
        self.splitcomm = mpicomm.Split(color, key)
        self.splitcomm.Set_name('MPILevel2-WorldRank:' + str(self.worldrank))
        print('SPOTPY/MPI ({}) - size: {}, rank: {}\n'.format(self.splitcomm.name, self.splitcomm.Get_size(), self.splitcomm.Get_rank()))

        self.params = [spotpy.parameter.Uniform('er',low=6, high=7, optguess=6),
                       spotpy.parameter.Uniform('sigma',low=0.001, high=0.1, optguess=0.001),
                       spotpy.parameter.Uniform('depth',low=0.15, high=0.25, optguess=0.25),
                       spotpy.parameter.Uniform('radius',low=0.04, high=0.12, optguess=0.05)]

        self.curdir = os.getcwd()
        filenameobs = 'cylinder_Bscan_2D_obs_merged.out'
        self.modelrxnumber = 1
        self.modelrxcomponent = 'Ez'
        outputdata, dt = get_output_data(self.curdir + os.sep + filenameobs, self.modelrxnumber, self.modelrxcomponent)
        self.evals = outputdata.flatten().tolist()

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        x = np.array(vector)

        # Create model input file from parameters
        modeltitle = 'cylinder_Bscan_2D'
        traces = 4
        inputfilebase = ['#title: ' + modeltitle, '#domain: 0.600 0.600 0.005', '#dx_dy_dz: 0.005 0.005 0.005', '#time_window: 8e-9', '#waveform: gaussiandotnorm 1 1e9 mypulse', '#hertzian_dipole: z 0.100 0.500 0 mypulse', '#rx: 0.200 0.500 0', '#src_steps: 0.100 0 0', '#rx_steps: 0.100 0 0']
        material = ['#material: {} {} 1 0 half_space\n'.format(x[0], x[1])]
        box = ['#box: 0 0 0 0.600 0.500 0.005 half_space']
        cylinder = ['#cylinder: 0.300 {} 0 0.300 {} 0.005 {} free_space'.format(x[2], x[2], x[3])]
        inputfilecomplete = inputfilebase + material + box + cylinder

        # Write input file
        inputfilenamebase = self.curdir + os.sep + modeltitle
        inputfilename = inputfilenamebase + str(self.worldrank) + '.in'
        with open(inputfilename, 'w') as f:
            for line in inputfilecomplete:
                f.write('{}\n'.format(line))

        # Run gprMax model(s)
        gprMax(inputfilename, n=traces, mpi=traces+1, mpicomm=self.splitcomm) # Add -gpu option if desired

        # Merge A-scans to B-scan and then remove A-scan files
        merge_files(inputfilenamebase + str(self.worldrank), removefiles=True)

        # Load gprMax result(s)
        outputfilename = inputfilenamebase + str(self.worldrank) + '_merged.out'
        outputdata, dt = get_output_data(outputfilename, self.modelrxnumber, self.modelrxcomponent)
        simulations = outputdata.flatten().tolist()

        # Remove input file and merged output file
        os.remove(inputfilename)
        os.remove(outputfilename)

        return simulations

    def evaluation(self):
        return self.evals

    def objectivefunction(self, simulation, evaluation, params=None):
        objectivefunction= -spotpy.objectivefunctions.rmse(evaluation,simulation)
        return objectivefunction

    def __del__(self):
        # MPI Configuration for SCEUA Optimiser
        self.splitcomm.Disconnect()
