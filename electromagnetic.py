import numpy as np

class electromagnetic:
    def __init__(self, main_class):
        self.calculate_E(main_class)
        if main_class.use_orbital:
            self.calculate_L(main_class)
        if main_class.use_spin:
            self.calculate_S(main_class)
        if main_class.use_xct:
            self.calculate_xct(main_class)


    def calculate_E(self, main_class):
        nk = int(main_class.nk)
        E = main_class.noeh_dipole[:,:,:,:]
        energy = main_class.energy_dft
        newE = np.zeros_like(E)

        for ik in range(nk):
            for ib1 in range(main_class.nv+main_class.nc):
                for ib2 in range(main_class.nv + main_class.nc):
                    energy_diff_for_cancel_diple = energy[ik, ib1] - energy[
                        ik, ib2]
                    if np.abs(energy_diff_for_cancel_diple)>0.0000001:
                        energy_diff_for_cancel_diple_inv = 1/energy_diff_for_cancel_diple
                    else:
                        energy_diff_for_cancel_diple_inv = 0

                    newE[ik,ib1,ib2,:] = E[ik,ib1,ib2,:]*energy_diff_for_cancel_diple_inv
        main_class.E_kvc = newE[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]
        print('finish calculating E_kvc')
        return 0
    def calculate_L(self, main_class):
        L = main_class.orbit
        energy = main_class.energy_dft
        newL = np.zeros_like(L)


        for ik in range(main_class.nk):
            for iv in range(main_class.nv+main_class.nc):
                for ic in range(main_class.nv + main_class.nc):
                    energy_diff_for_cancel_diple = energy[ik, iv] - energy[
                        ik, ic]
                    if np.abs(energy_diff_for_cancel_diple)>0.0000001:
                        energy_diff_for_cancel_diple_inv = 1/energy_diff_for_cancel_diple
                    else:
                        energy_diff_for_cancel_diple_inv = 0

                    newL[ik,iv,ic,:] = L[ik,iv,ic,:]*energy_diff_for_cancel_diple_inv
        if main_class.hermitian_convert:
            newL2 = np.zeros_like(L)
            for ik in range(main_class.nk):
                for iv in range(main_class.nv+main_class.nc):
                    for ic in range(main_class.nv+main_class.nc):
                        newL2[ik, iv, ic, 0] = (newL[ik, iv, ic, 0]+ np.conj(newL[ik, ic, iv, 0]))/2
                        newL2[ik, iv, ic, 1] = (newL[ik, iv, ic, 1]+ np.conj(newL[ik, ic, iv, 1]))/2
                        newL2[ik, iv, ic, 2] = (newL[ik, iv, ic, 2]+ np.conj(newL[ik, ic, iv, 2]))/2
            main_class.L_kvc = newL2[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]
        else:
            main_class.L_kvc = newL[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]
        print('finish calculating L_kvc')
        return 0
    def calculate_S(self, main_class):
        S = main_class.spin
        energy = main_class.energy_dft
        newS = np.zeros_like(S)

        for ik in range(main_class.nk):
            for iv in range(main_class.nv+main_class.nc):
                for ic in range(main_class.nv + main_class.nc):
                    energy_diff_for_cancel_diple = energy[ik, iv] - energy[
                        ik, ic]
                    if np.abs(energy_diff_for_cancel_diple)>0.0000001:
                        energy_diff_for_cancel_diple_inv = 1/energy_diff_for_cancel_diple
                    else:
                        energy_diff_for_cancel_diple_inv = 0

                    newS[ik,iv,ic,:] = S[ik,iv,ic,:]*energy_diff_for_cancel_diple_inv
        main_class.S_kvc = newS[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]
        print('finish calculating S_kvc')
        return 0
    def calculate_xct(self, main_class):
        idx = list(range(main_class.nv - 1, -1, -1))
        nk = main_class.nk
        nk_sub = int(main_class.nk/7)
        inds = np.ix_(range(main_class.nk), idx, range(main_class.nc), range(3))
        E_temp = main_class.E_kvc[inds]
        if main_class.use_orbital:
            L_temp = main_class.L_kvc[inds]
        if main_class.use_spin:
            S_temp = main_class.S_kvc[inds]

        a = np.einsum('kvcs,kvcd->sd', main_class.avck[0:nk_sub,:,:,:], E_temp[0:nk_sub,:,:,:])
        b = np.einsum('kvcs,kvcd->sd', main_class.avck[:,:,:,:], E_temp[:,:,:,:])
        factor_test = b/a
        factor = np.abs(b/a)
        # test1 = np.abs(main_class.avck[512*0:512*1,:,:,:])
        # test2 = np.abs(main_class.avck[512*1:512*2,:,:,:])
        # test3 = np.abs(main_class.avck[512*2:512*3,:,:,:])
        # test4 = (test3-test2)/0.0125
        main_class.ME = a*factor
        ###for test
        E_kvc_length = main_class.noeh_dipole_length[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]
        inds_2 = np.ix_(range(nk_sub), idx, range(main_class.nc), range(3))
        E_temp_2 = E_kvc_length[inds_2]
        c = np.einsum('kvcs,kvcd->sd', main_class.avck[0:nk_sub,:,:,:], E_temp_2)
        main_class.ME = c * factor/2



        if main_class.use_orbital:
            main_class.MM = np.einsum('kvcs,kvcd->sd', main_class.avck[0:nk,:,:,:], L_temp)
        if main_class.use_spin:
            main_class.MS = np.einsum('kvcs,kvcd->sd', main_class.avck[0:nk,:,:,:], S_temp)

        print('finish calculating xct')
        return 0

