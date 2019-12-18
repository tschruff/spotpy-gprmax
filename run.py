"""

"""
from mpi4py import MPI
from gprMax._version import __version__
from gprMax.mpi import MPIExecutor
from gprMax.model_build_run import run_model
from itertools import cycle
import numpy as np
import h5py
import spotpy
import os
import logging
import argparse

try:
    import pycuda.driver as drv
    from pycuda.driver import RuntimeError as CUDAError
    
    try:
        drv.init()  # might raise CUDAError if no CUDA devices where found
    except CUDAError as e:
        raise ImportError(str(e))

    GPU_COUNT = drv.Device.count()
    HAS_CUDA = True

    def gpu_device(bus_id):
        return drv.Device(bus_id)

except ImportError:
    drv = None
    GPU_COUNT = 0
    HAS_CUDA = False

    def gpu_device(bus_id):
        assert bus_id is None
        return bus_id

_log = logging.getLogger('gprMax.setup')


class Setup(object):

    def __init__(self, gpr, ntraces, ncomplex, datadir, datafile, geometry_only=False, geometry_fixed=False,
                 write_processed=False, delete_gpr_output=True, namespace=None, gpus=None, gpu_fallback='cpu'):

        if gpus is not None:
            assert HAS_CUDA and GPU_COUNT

        self.gpr = gpr
        self.ntraces = ntraces
        self.complex = ncomplex
        self.datadir = datadir
        self.datafile = datafile
        self.geometry_only = geometry_only
        self.geometry_fixed = geometry_fixed
        self.write_processed = write_processed
        self.delete_gpr_output = delete_gpr_output
        self.namespace = namespace or {}

        self.params = [spotpy.parameter.Uniform('er', low=6, high=7, optguess=6),
                       spotpy.parameter.Uniform('sigma', low=0.001, high=0.1, optguess=0.001),
                       spotpy.parameter.Uniform('depth', low=0.15, high=0.25, optguess=0.25),
                       spotpy.parameter.Uniform('radius', low=0.04, high=0.12, optguess=0.05)]

        filenameobs = 'cylinder_Bscan_2D_obs_merged.out'
        self.modelrxnumber = 1
        self.modelrxcomponent = 'Ez'

        outputdata, dt = self.__get_output_data(os.path.join(self.datadir, filenameobs),
                                                self.modelrxnumber, self.modelrxcomponent)

        self.evals = outputdata.flatten().tolist()

        if self.gpr is not None:
            num_workers = self.gpr.size - 1
            if gpus is None:
                self.gpus = [None] * num_workers
            elif len(self.gpus) >= num_workers:
                self.gpus = self.gpus[:num_workers]
            elif gpu_fallback == 'gpu':
                self.gpus = (((num_workers // len(gpus)) + 1) * gpus)[:num_workers]
            elif gpu_fallback == 'cpu':
                self.gpus = gpus + [None] * (num_workers - len(gpus))
            else:
                raise ValueError(f'invalid gpu fallback {gpu_fallback}')

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def __create_input(self, filename, vector):
        """

        """
        er, sigma, depth, radius = tuple(vector)
        # Create model input file from parameters
        modeltitle, _ = os.path.splitext(self.datafile)

        inputfilebase = ['#title: ' + modeltitle, '#domain: 0.600 0.600 0.005', '#dx_dy_dz: 0.005 0.005 0.005',
                         '#time_window: 8e-9', '#waveform: gaussiandotnorm 1 1e9 mypulse',
                         '#hertzian_dipole: z 0.100 0.500 0 mypulse', '#rx: 0.200 0.500 0',
                         '#src_steps: 0.100 0 0', '#rx_steps: 0.100 0 0']

        material = [f'#material: {er} {sigma} 1 0 half_space\n']
        box = ['#box: 0 0 0 0.600 0.500 0.005 half_space']
        cylinder = [f'#cylinder: 0.300 {depth} 0 0.300 {depth} 0.005 {radius} free_space']
        data = inputfilebase + material + box + cylinder

        with open(filename, 'w') as fh:
            fh.write('\n'.join(data))

    def __merge_files(self, infiles, outfile, remove_infiles=False):
        modelruns = len(infiles)

        # Combined output file
        fout = h5py.File(outfile, 'w')

        # Add positional data for rxs
        for i, infile in enumerate(infiles):
            fin = h5py.File(infile, 'r')
            nrx = fin.attrs['nrx']

            # Write properties for merged file on first iteration
            if i == 0:
                fout.attrs['Title'] = fin.attrs['Title']
                fout.attrs['gprMax'] = __version__
                fout.attrs['Iterations'] = fin.attrs['Iterations']
                fout.attrs['dt'] = fin.attrs['dt']
                fout.attrs['nrx'] = fin.attrs['nrx']
                for rx in range(1, nrx + 1):
                    path = '/rxs/rx' + str(rx)
                    grp = fout.create_group(path)
                    availableoutputs = list(fin[path].keys())
                    for output in availableoutputs:
                        grp.create_dataset(output, (fout.attrs['Iterations'], modelruns),
                                           dtype=fin[path + '/' + output].dtype)

            # For all receivers
            for rx in range(1, nrx + 1):
                path = '/rxs/rx' + str(rx) + '/'
                availableoutputs = list(fin[path].keys())
                # For all receiver outputs
                for output in availableoutputs:
                    fout[path + '/' + output][:, i] = fin[path + '/' + output][:]

            fin.close()

        fout.close()

        if remove_infiles:
            for infile in infiles:
                os.remove(infile)

    def __get_output_data(self, filename, rxnumber, rxcomponent):
        """Gets B-scan output data from a model.

        Args:
            filename (string): Filename (including path) of output file.
            rxnumber (int): Receiver output number.
            rxcomponent (str): Receiver output field/current component.

        Returns:
            outputdata (array): Array of A-scans, i.e. B-scan data.
            dt (float): Temporal resolution of the model.
        """

        with h5py.File(filename, 'r') as f:

            nrx = f.attrs['nrx']
            dt = f.attrs['dt']

            # Check there are any receivers
            if nrx == 0:
                raise ValueError(f'No receivers found in {filename}')

            path = '/rxs/rx' + str(rxnumber) + '/'
            availableoutputs = list(f[path].keys())

            # Check if requested output is in file
            if rxcomponent not in availableoutputs:
                raise ValueError('{} output requested to plot, but the available '
                                 'output for receiver 1 is {}'.format(rxcomponent, ', '.join(availableoutputs)))

            outputdata = f[path + '/' + rxcomponent]
            outputdata = np.array(outputdata)

        return outputdata, dt

    def simulation(self, vector):

        assert self.gpr is not None

        # Create input file object
        modeltitle = self.datafile
        filenamebase = os.path.join(self.datadir, modeltitle)
        filenamebase = filenamebase + f'_{self.complex:d}'
        infile = filenamebase + '.in'

        self.__create_input(infile, vector)

        jobs = []
        for j, gpu in zip(range(self.ntraces), cycle(self.gpus)):

            model_args = argparse.Namespace(**{
                'geometry_only': self.geometry_only,
                'geometry_fixed': self.geometry_fixed,
                'write_processed': self.write_processed,
                'task': False,
                'restart': False,
                'gpu': gpu_device(gpu)
            })

            jobs.append({
                'args': model_args,
                'inputfile': infile,
                'currentmodelrun': j + 1,
                'modelend': self.ntraces,
                'numbermodelruns': self.ntraces,
                'usernamespace': self.namespace.copy()
            })

        _log.info(f'Running GPRMax with parameters {vector}.')

        self.gpr.submit(jobs)

        _log.info('Merging GPRMax results.')

        if self.ntraces == 1:
            outfiles = [filenamebase + '.out']
        else:
            outfiles = [filenamebase + str(n+1) + '.out' for n in range(self.ntraces)]

        outfile = filenamebase + '.out'

        # Merge output files (A-scans to B-scan) and then remove A-scan output files
        self.__merge_files(outfiles, outfile, remove_infiles=self.delete_gpr_output)

        _log.info('Processing GPRMax results.')

        # Load gprMax result(s)
        outputdata, dt = self.__get_output_data(outfile, self.modelrxnumber, self.modelrxcomponent)
        simulations = outputdata.flatten().tolist()

        if self.delete_gpr_output:
            _log.info(f'Cleaning up GPRMax input and output.')
            os.remove(infile)
            os.remove(outfile)

        _log.info('Finished simulation.')

        return simulations

    def evaluation(self):
        return self.evals

    def objectivefunction(self, simulation, evaluation, params):
        assert self.gpr is None, 'objectivefunction must not be called on a spotpy worker'
        # FIXME: why does this fail?
        # ls, le = len(simulation), len(evaluation)
        # assert ls == le, f'len(simulation) = {ls}, len(evaluation) = {le}'
        return -spotpy.objectivefunctions.rmse(evaluation, simulation)


def main(args=None):

    import argparse
    import os

    parser = argparse.ArgumentParser('spotpy-gprmax')
    parser.add_argument('num_spotpy_workers', type=int,
                        help='The number of spotpy workers.')
    parser.add_argument('-m', '--model', type=str, default='cylinder_Bscan_2D',
                        help='The model name.')
    parser.add_argument('-t', '--ntraces', type=int, default=1,
                        help='The number of traces (A-scans) per spotpy parameter set (function evaluation).')
    parser.add_argument('-r', '--nrep', type=int, default=10,
                        help='The maximum number of function evaluations allowed during optimization.')
    parser.add_argument('-c', '--ncplx', type=int, default=5,
                        help='The number of spotpy complexes. Take more than the number of analysed parameters.')
    parser.add_argument('-l', '--logfile', action='store_true', default=False,
                        help='Create logfiles.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Increase logging output.')
    parser.add_argument('-f', '--data-folder', type=str, default=os.getcwd(),
                        help='The folder that contains the input file. Output will also pe placed here.')
    parser.add_argument('-d', '--delete-gpr-output', action='store_true', default=False,
                        help='Delete (temporary) gprMax output files.')
    parser.add_argument('-g', '--gpu', action='store_true', default=False,
                        help='Enable GPU mode.')
    parser.add_argument('-e', '--log-env', action='store_true', default=False,
                        help='Write all environment variables to the root log.')

    args = parser.parse_args(args)

    host = MPI.Get_processor_name()
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    logger = logging.getLogger('gprMax')
    level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(level)

    if args.logfile:
        mh = logging.FileHandler(f"log_{rank}.txt", mode='w')
        mh.setLevel(level)
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
        mh.setFormatter(formatter)
        logger.addHandler(mh)

    if args.log_env and rank == 0:
        logger.info('Environment:, {}'.format(os.environ))

    num_workers = args.num_spotpy_workers

    if rank == 0:
        logger.info(f'Running spotpy-gprmax on {host} with {size} processes and {num_workers} spotpy workers')
        logger.info(f'CUDA available: {HAS_CUDA}')
        logger.info(f'Number of CUDA devices: {GPU_COUNT}')

    # ==================================
    # SETUP MPI COMMUNICATORS
    # ==================================

    required_size = 2 * num_workers + 1
    if size < required_size:
        logger.error(f'Not enough processes allocated! At least {required_size} '
                     f'required but only {size} allocated.')
        comm.Abort(1)

    # split into spotpy and gprMax group
    if rank <= num_workers:
        color = 0  # spotpy master/worker group
    elif num_workers == 1:
        color = 1  # gprmax worker group
    else:
        color = (rank - 1) % (size - (num_workers + 1)) + 1
    
    logger.debug(f'Color on rank {rank}: {color}')

    colors = np.array(comm.allgather(color))
    if rank == 0:
        logger.info(f'MPI group colors per rank: {colors}')

    if np.max(colors) < num_workers:
        logger.error('Invalid configuration. Not enough processes allocated.')
        comm.Abort(1)

    # setup MPI communicators
    wgroup = comm.group
    comms = [MPI.COMM_NULL] * (num_workers + 1)
    for i in range(0, num_workers + 1):

        if i == 0:
            # spotpy communicator
            ranks, = np.nonzero(colors == 0)
        else:
            # spotpy-gprMax communicators
            ranks = np.concatenate([[i], np.nonzero(colors == i)[0]])

        group = wgroup.Incl(ranks)
        icomm = comm.Create_group(group)
        if icomm != MPI.COMM_NULL:
            icomm.name = f'COMM-{i:d}'
            irank = icomm.rank
            if irank == 0:
                logger.debug(f'Size of {icomm.name}: {icomm.size}')
        comms[i] = icomm

    logger.info(f'Comms on rank {rank:d}: %s' % str([comm.name if comm else 'COMM_NULL' for comm in comms]))

    color = colors[rank]
    icomm = comms[color]
    # rank within the group
    irank = icomm.rank

    is_spotpy_master = color == 0 and irank == 0
    is_spotpy_worker = is_gpr_master = color == 0 and irank != 0
    is_gpr_worker = color != 0

    # ==================================
    # SETUP GPU
    # ==================================

    if args.gpu:
        if not HAS_CUDA:
            logger.warning('GPU support disabled because pycuda is not installed')
            gpus = None
        elif GPU_COUNT == 0:
            logger.warning('GPU support disabled because no CUDA devices available')
            gpus = None
        else:
            if is_spotpy_master:
                # no need for GPU support on spotpy master
                gpus = None
            elif is_gpr_master:
                # node_id = os.environ.get('SLURM_NODEID')
                # NOTE: CUDA_VISIBLE_DEVICES returns only GPUs on the LOCAL node
                # gpus = tuple(np.fromstring(os.environ.get('CUDA_VISIBLE_DEVICES'), dtype=int, sep=','))
                logger.warning('GPU support not yet implemented. Falling back to CPU mode.')
                gpus = None
            else:
                gpus = None
    else:
        gpus = None

    logger.info(f'GPU: {gpus}')

    # ==================================
    # SETUP MPI EXECUTOR
    # ==================================

    if is_spotpy_master:
        logger.info('=== SPOTPY MASTER ===')
        spotpy_comm = comms[0]
        gpr = None
        terminate = None
    elif is_spotpy_worker:
        logger.info('=== SPOTPY WORKER / GPRMAX MASTER ===')
        spotpy_comm = comms[0]
        gpr_comm = comms[irank]
        gpr = MPIExecutor(run_model, comm=gpr_comm)
        gpr.start()
        terminate = gpr.join
    elif is_gpr_worker:
        logger.info('=== GPRMAX WORKER ===')
        spotpy_comm = comms[0]
        gpr_comm = comms[color]
        gpr = MPIExecutor(run_model, comm=gpr_comm)
        terminate = None
        gpr.start()
    else:
        logger.error(f'invalid group on rank {rank}')
        comm.Abort(1)
        raise RuntimeError('')  # just to keep pylint from complaining

    # only spotpy master and workers
    # gpr workers are already waiting for jobs
    if color == 0:

        basepath = os.path.join(args.data_folder, args.model)

        setup = Setup(gpr=gpr, ntraces=args.ntraces,
                      ncomplex=irank, datadir=args.data_folder,
                      datafile=args.model, delete_gpr_output=args.delete_gpr_output, gpus=gpus)

        logger.info('Initializing spotpy SCEUA complexes')
        sampler = spotpy.algorithms.sceua(setup, dbname=basepath, dbformat='csv',
                                          parallel='mpi', parallel_kwargs=dict(mpicomm=spotpy_comm,
                                                                               on_worker_terminate=terminate))

        logger.info('Performing spotpy sampling')
        try:
            sampler.sample(args.nrep, ngs=args.ncplx)
        except Exception as e:
            logger.exception(str(e))
            spotpy_comm.Abort(1)

        # NOTE: spotpy workers will never reach this because spotpy kills them
        #       after the work has been completed (which is not nice at all)

        logger.info('Finished spotpy sampling.')

        # only executed on the spotpy master
        if irank == 0:
            # actually it is not necessary to limit this to the spotpy master
            # since the spotpy workers will be dead by now and the gprmax workers
            # will never reach this since color == 0
            # keep this purely for illustrative reasons

            # retrieve results from spotpy master
            result = sampler.getdata()

    logger.info('Finished spotpy.')


if __name__ == '__main__':
    main()
