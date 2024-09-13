import numpy as np
import reader
import electromagnetic
import h5py as h5
import length_gauge_mod

class main_class:
    '''
    This is the main class where most parameters and data store
    '''

    def __init__(self, nc, nv, nc_in_file, nv_in_file ,hovb, nxct, input_folder, W, eta , use_eqp, energy_shift, eps1_correction, use_xct, use_spin, use_orbital, hermitian_convert, polarization, molecule, plot_wavelength, cd_exp_unit,exp_unit_converter,length_gauge,dk):
        """
        intialize main_class from input.py and all the input files
        """
        self.nc = nc #number of conduction bands in eigenvectors.h5
        self.nv = nv  #number of valence bands in eigenvectors.h5
        self.nb = nv+nc
        self.hovb = hovb # index of the highest occupied band
        self.nxct = nxct # number of exciton states
        self.input_folder = input_folder #address of input folder
        self.W = W #energy range
        self.eta = eta #broadening coefficient
        self.use_eqp = use_eqp #use eqp correction or not
        self.energy_shift = energy_shift
        self.eps1_correction = eps1_correction
        self.nc_in_file = nc_in_file
        self.nv_in_file = nv_in_file
        self.use_xct = use_xct
        self.use_spin = use_spin
        self.use_orbital = use_orbital
        self.hermitian_convert = hermitian_convert
        self.polarization = polarization
        self.molecule = molecule
        self.plot_wavelength = plot_wavelength
        self.cd_exp_unit = cd_exp_unit
        self.exp_unit_converter = exp_unit_converter
        self.length_gauge = length_gauge
        self.dk = dk

        reader.reader(self)
        if self.length_gauge:
            length_gauge_mod.wfn_derivative(self)
            length_gauge_mod.Avck_derivative2(self)
            length_gauge_mod.Ddipole(self)
        electromagnetic.electromagnetic(self)

