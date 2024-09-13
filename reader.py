import numpy as np
from mpi import MPI, comm, size, rank
import h5py as h5
import math_function

class reader:
    """
    read data from input files
    """
    def __init__(self, main_class):
        self.read_header(main_class)
        self.read_dipole(main_class)
        if main_class.use_spin:
            self.read_spin(main_class)
        if main_class.use_orbital:
            self.read_orbit(main_class)
        if main_class.use_xct:
            self.read_xct(main_class)
        if main_class.use_eqp:
            self.read_eqp(main_class)

    def read_header(self, main_class):
        input_file = main_class.input_folder + 'wfn.h5'
        f = h5.File(input_file, 'r')
        main_class.energy_dft = f['mf_header/kpoints/el'][0, :, main_class.hovb - main_class.nv: main_class.hovb + main_class.nc]
        main_class.volume = f['mf_header/crystal/celvol'][()]
        main_class.rk = f['mf_header/kpoints/rk'][()]
        main_class.nk = f['mf_header/kpoints/nrk'][()]
        main_class.spinor = f['mf_header/kpoints/nspinor'][()]
        main_class.a = f['/mf_header/crystal/alat'][()]*f['/mf_header/crystal/avec'][()]*0.52917
        main_class.bdot = f['/mf_header/crystal/bdot'][()]
        f.close()
        print('finish reading wfn_header.h5')
        return 0

    def read_dipole(self, main_class):
        input_file = main_class.input_folder + 'dipole_matrix.h5'
        f = h5.File(input_file, 'r')
        #main_class.temp = f['dipole'][()]
        main_class.noeh_dipole = f['dipole'][:, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc,
                                 main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]

        f.close()
        print('finish reading dipole_matrix.h5')
        return 0

    def read_spin(self, main_class):
        input_file = main_class.input_folder + 'spin_matrix.h5'
        f = h5.File(input_file, 'r')

        main_class.spin = f['spin'][:, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc,
                                 main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]

        f.close()
        print('finish reading spin_matrix.h5')
        return 0

    def read_orbit(self, main_class):
        input_file = main_class.input_folder + 'orbit_matrix.h5'
        f = h5.File(input_file, 'r')

        main_class.orbit = 1*f['orbit'][:, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc,
                                 main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]

        f.close()
        print('finish reading orbit_matrix.h5')
        return 0

    def read_xct(self, main_class):
        input_file = main_class.input_folder + 'eigenvectors.h5'
        f = h5.File(input_file, 'r')
        avck = f['exciton_data/eigenvectors'][()]
        main_class.excited_energy = f['exciton_data/eigenvalues'][0:main_class.nxct]
        f.close()

        avck = np.transpose(avck, (
            0, 1, 2, 4, 3, 5, 6))  # eigenvectors in the h5 file is [..., c , v ...], we convert it to [..., v , c ...]
        avck = avck[0, 0:main_class.nxct, :, :, :, 0, 0] + 1j * avck[0, 0:main_class.nxct, :, :, :, 0, 1]
        avck = np.transpose(avck, (1, 2, 3, 0))
        print('finish reading Acvk from eigenvectors.h5')
        main_class.avck = avck

        return 0

    def read_eqp(self, main_class):
        input_file = main_class.input_folder + 'eqp.dat'

        data = np.loadtxt(input_file)

        eqp_corr = np.zeros([main_class.nk, main_class.nc + main_class.nv])
        for ik in range(main_class.nk):
            for ib in range(main_class.nc + main_class.nv):
                eqp_corr[ik, ib] = data[ik * (main_class.nc + main_class.nv + 1) + ib + 1, 3] - data[
                    ik * (main_class.nc + main_class.nv + 1) + ib + 1, 2]
        print('finish loading quasi-particle energy from {0:s}'.format('eqp.dat'))
        main_class.eqp_corr = eqp_corr
        return 0

