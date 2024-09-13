import main_class
import optical
import sys
input_folder = './Si/'
sys.path.insert(1,input_folder)
from CD_h5_inp import *


if __name__ == '__main__':
    main_class = main_class.main_class(nc, nv, nc_in_file, nv_in_file, hovb, nxct, input_folder, W, eta , use_eqp, energy_shift, eps1_correction, use_xct, use_spin, use_orbital, hermitian_convert, polarization, molecule, plot_wavelength, cd_exp_unit, exp_unit_converter, length_gauge, dk)

    #optical.calculate_absorption_eh(main_class)
    #optical.calculate_absorption_noeh(main_class)
    optical.calculate_epsR_epsL_eh(main_class)
    #optical.calculate_epsR_epsL_noeh(main_class)

    #optical.CPL_eh(main_class)

