import numpy as np
from math_function import *
import matplotlib.pyplot as plt

def calculate_absorption_eh(main_class):
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction
    volume = main_class.volume
    nk = main_class.nk
    W = main_class.W
    EE = main_class.ME
    eta = main_class.eta

    pref = 16.0 * np.pi ** 2 / volume / nk/main_class.spinor*4
    RYD = 13.6057039763  # W=W/RYD
    eps2 = np.zeros_like(W)
    eps1 = np.zeros_like(W)

    if main_class.polarization == 'x':
        E1 = (EE[:, 0])
    elif main_class.polarization == 'y':
        E1 = (EE[:, 1])
    elif main_class.polarization == 'z':
        E1 = (EE[:, 2])
    elif main_class.polarization == 'L':
        E1 = ((EE[:, 0])+1j*(EE[:, 1]))/np.sqrt(2)
    elif main_class.polarization == 'R':
        E1 = ((EE[:, 0])-1j*(EE[:, 1]))/np.sqrt(2)


    if main_class.molecule:
        for s in range(main_class.nxct):
            energyDif = main_class.excited_energy[s] + energy_shift
            eps2 += (np.abs(EE[:, 0]) ** 2+np.abs(EE[:, 1]) ** 2+np.abs(EE[:, 2]) ** 2)/3 * delta_gauss(W / RYD, energyDif / RYD, eta / RYD)
            eps1 += (np.abs(EE[:, 0]) ** 2+np.abs(EE[:, 1]) ** 2+np.abs(EE[:, 2]) ** 2)/3 * (delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)) * (
                        energyDif / RYD - W / RYD) / eta * RYD
    else:
        for s in range(main_class.nxct):
            energyDif = main_class.excited_energy[s] + energy_shift
            eps2 += np.abs(E1[s]) ** 2 * delta_gauss(W / RYD, energyDif / RYD, eta / RYD)
            eps1 += np.abs(E1[s]) ** 2 * (delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)) * (
                        energyDif / RYD - W / RYD) / eta * RYD



    eps2 *= pref
    eps1 = pref * eps1 + 1 + eps1_correction

    plt.figure()
    if main_class.plot_wavelength:
        plt.plot(1240/W, eps2, 'r', label='eps2')
        #plt.plot(1240/W, eps1, 'b', label='eps1')
    else:
        plt.plot(W, eps2, 'r', label='eps2')
        #plt.plot(W, eps1, 'b', label='eps1')

    plt.legend()
    plt.show()

    data = np.array([W, eps2, eps1])
    np.savetxt(main_class.input_folder+'absp.dat', data.T)
    return 0
def calculate_absorption_noeh(main_class):
    RYD = 13.6057039763
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction
    E_kvc = main_class.E_kvc
    nk = main_class.nk
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    W = main_class.W
    eta = main_class.eta
    volume = main_class.volume
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr = main_class.eqp_corr

    pref = 16.0 * np.pi**2/volume/nk/main_class.spinor*4 # 4 comes from v = p/m ~ 2p

    eps2 = np.zeros_like(W)
    eps1 = np.zeros_like(W)

    if main_class.polarization == 'x':
        E1 = (E_kvc[:, :, :, 0])
    elif main_class.polarization == 'y':
        E1 = (E_kvc[:, :, :, 1])
    elif main_class.polarization == 'z':
        E1 = (E_kvc[:, :, :, 2])
    elif main_class.polarization == 'L':
        E1 = ((E_kvc[:, :, :, 0])+1j*(E_kvc[:, :, :, 1]))/np.sqrt(2)
    elif main_class.polarization == 'R':
        E1 = ((E_kvc[:, :, :, 0])-1j*(E_kvc[:, :, :, 1]))/np.sqrt(2)

    if main_class.molecule:
        for ik in range(nk):
            for iv in range(nv):
                for ic in range(nc):
                    energyDif = energy_dft[ik,ic+nv]-energy_dft[ik,iv]+energy_shift/RYD
                    if use_eqp:
                        energyDif2 =energyDif + eqp_corr[ik,nv+ic]/RYD-eqp_corr[ik,iv]/RYD
                    else:
                        energyDif2 =energyDif

                    eps2 += (np.abs((E_kvc[:, :, :, 0]))**2+np.abs((E_kvc[:, :, :, 1]))**2+np.abs((E_kvc[:, :, :, 2]))**2)/3 * (delta_gauss(W/RYD, energyDif2, eta/RYD))
                    eps1 += (np.abs((E_kvc[:, :, :, 0]))**2+np.abs((E_kvc[:, :, :, 1]))**2+np.abs((E_kvc[:, :, :, 2]))**2)/3 * (
                        delta_lorentzian(W / RYD, energyDif2, eta / RYD))*(energyDif2-W/RYD)/eta * RYD
    else:
        for ik in range(nk):
            for iv in range(nv):
                for ic in range(nc):
                    energyDif = energy_dft[ik, ic + nv] - energy_dft[ik, iv] + energy_shift / RYD
                    if use_eqp:
                        energyDif2 = energyDif + eqp_corr[ik, nv + ic] / RYD - eqp_corr[ik, iv] / RYD
                    else:
                        energyDif2 = energyDif

                    eps2 += np.abs(E1[ik, iv, ic]) ** 2 * (delta_gauss(W / RYD, energyDif2, eta / RYD))
                    eps1 += np.abs(E1[ik, iv, ic]) ** 2 * (
                        delta_lorentzian(W / RYD, energyDif2, eta / RYD)) * (energyDif2 - W / RYD) / eta * RYD


    eps2 *= pref
    eps1 = pref*eps1 + 1 + eps1_correction

    plt.figure()
    if main_class.plot_wavelength:
        plt.plot(1240 / W, eps2, 'r', label='eps2')
        #plt.plot(1240 / W, eps1, 'b', label='eps1')
    else:
        plt.plot(W, eps2, 'r', label='eps2')
        #plt.plot(W, eps1, 'b', label='eps1')
    plt.show()

    data = np.array([W, eps2, eps1])
    np.savetxt(main_class.input_folder+'absp0.dat', data.T)

    return

def calculate_epsR_epsL_eh(main_class):
    energy_shift=main_class.energy_shift
    volume = main_class.volume
    nk = main_class.nk
    W = main_class.W
    EE = main_class.ME
    MM = main_class.MM
    if main_class.use_spin:
        MM+=main_class.MS
    eta = main_class.eta

    pref = 16.0 * np.pi ** 2 / volume /nk/main_class.spinor*4
    RYD = 13.6057039763  # W=W/RYD
    light_speed = 274
    epsilon_r = 1
    eps2 = np.zeros_like(W)
    CD = np.zeros_like(W)


    E1 = ((EE[:, 0])+1j*(EE[:, 1]))/np.sqrt(2)



    for s in range(main_class.nxct):
        energyDif = main_class.excited_energy[s]+energy_shift
        eps2 += np.abs(E1[s]) ** 2 \
                   * delta_gauss(W / RYD, energyDif/RYD, eta / RYD)
        CD += np.real(
            EE[s, 0] * np.conj(MM[s, 0]) + EE[s, 1] * np.conj(MM[s, 1])) \
             * W / RYD / light_speed / epsilon_r * delta_gauss(W / RYD, energyDif/RYD, eta / RYD) * 2



    eps2 *= pref

    CD*=pref
    if main_class.cd_exp_unit:
        CD*=main_class.exp_unit_converter


    plt.figure()
    if main_class.plot_wavelength:
        plt.plot(1240/W, CD, 'r', label='L-R')
    else:
        plt.plot(W, CD, 'r', label='L-R')
    plt.legend()
    plt.show()

    plt.figure()
    if main_class.plot_wavelength:
        plt.plot(1240/W, eps2, 'r', label='ep2')
    else:
        plt.plot(W, eps2, 'r', label='ep2')
    plt.legend()
    plt.show()

    data = np.array([W, eps2, CD])
    np.savetxt(main_class.input_folder+'CD.dat', data.T)

    return 0


def calculate_epsR_epsL_noeh(main_class):
    energy_shift=main_class.energy_shift
    volume = main_class.volume
    nk = main_class.nk
    if main_class.length_gauge:
        nk = int(nk/7)
    W = main_class.W
    E_kvc = main_class.E_kvc
    if main_class.length_gauge:
        L_kvc = main_class.L_kvc_length
    else:
        L_kvc = main_class.L_kvc
    if main_class.use_spin:
        L_kvc += main_class.S_kvc*2
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr= main_class.eqp_corr
    eta = main_class.eta



    pref = 16.0 * np.pi ** 2 / volume / nk/main_class.spinor*4
    RYD = 13.6057039763  # W=W/RYD
    light_speed = 274
    epsilon_r = 1
    eps2 = np.zeros_like(W)
    CD = np.zeros_like(W)


    E1 = (E_kvc[:, :, :, 0] + 1j * E_kvc[:, :, :, 1])/np.sqrt(2)



    for ik in range(nk):
        for iv in range(nv): #range(nv)
            for ic in range(nc): #range(nc)
                energyDif = energy_dft[ik, ic + nv] - energy_dft[ik, iv]+energy_shift/RYD
                if use_eqp:
                    energyDif2 = energyDif + eqp_corr[ik, nv + ic] / RYD - eqp_corr[ik, iv] / RYD
                else:
                    energyDif2 = energyDif
                eps2 += np.abs(E1[ik, iv, ic]) ** 2 * delta_gauss(W / RYD, energyDif2, eta / RYD)
                CD += np.real(E_kvc[ik,iv,ic, 0]*np.conj(L_kvc[ik,iv,ic,0])+E_kvc[ik,iv,ic, 1]*np.conj(L_kvc[ik,iv,ic,1]))\
                      * W / RYD / light_speed / epsilon_r*delta_gauss(W / RYD, energyDif2, eta / RYD)*2




    eps2 *= pref

    CD*=pref
    if main_class.cd_exp_unit:
        CD*=main_class.exp_unit_converter


    plt.figure()
    if main_class.plot_wavelength:
        plt.plot(1240/W, CD, 'r', label='L-R')
    else:
        plt.plot(W, CD, 'r', label='L-R')
    plt.legend()
    plt.show()

    plt.figure()
    if main_class.plot_wavelength:
        plt.plot(1240/W, eps2, 'r', label='eps2')
    else:
        plt.plot(W, eps2, 'r', label='eps2')
    plt.legend()
    plt.show()

    data = np.array([W, eps2, CD])
    np.savetxt(main_class.input_folder+'CD0.dat', data.T)

    return 0

def CPL_eh(main_class):

    energy_shift=main_class.energy_shift
    volume = main_class.volume
    nk = main_class.nk
    W = main_class.W
    EE = main_class.ME
    eta = main_class.eta

    pref = 16.0 * np.pi ** 2 / volume /nk/main_class.spinor*4
    RYD = 13.6057039763  # W=W/RYD
    RR = np.zeros_like(W)
    RL = np.zeros_like(W)


    E_L = ((EE[:, 0])+1j*(EE[:, 1]))/np.sqrt(2)
    E_R = ((EE[:, 0])-1j*(EE[:, 1]))/np.sqrt(2)

    osc_RR = np.abs(E_R*np.conj(E_R))**2
    osc_RL = np.abs(E_R*np.conj(E_L))**2



    for s in range(main_class.nxct):
        energyDif = main_class.excited_energy[s]+energy_shift


        RR += osc_RR[s]* delta_gauss(W / RYD, energyDif/RYD, eta / RYD)
        RL += osc_RL[s]* delta_gauss(W / RYD, energyDif/RYD, eta / RYD)
    RR*=pref
    RL*=pref

    plt.figure()
    if main_class.plot_wavelength:
        plt.plot(1240/W, RR, 'r', label='RR')
        plt.plot(1240 / W, RL, 'b', label='RL')
        plt.plot(1240 / W, RR-RL, 'k', label='RR-RL')
    else:
        plt.plot(W, RR, 'r', label='RR')
        plt.plot(W, RL, 'b', label='RL')
        plt.plot(W, RR - RL, 'k', label='RR-RL')
    plt.legend()
    plt.show()

    data = np.array([W, RR, RL])
    np.savetxt(main_class.input_folder+'CPL.dat', data.T)

    return 0

