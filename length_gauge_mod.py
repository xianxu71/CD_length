import numpy as np
import h5py as h5
def b123_to_xyz(a,v1,v2,v3):
    '''
    convert v in b1 b2 b3 direction to v in x y z direction
    '''
    V = np.inner(np.cross(a[0], a[1]), a[2])

    b1 = 2 * np.pi / V * (np.cross(a[1], a[2]))
    b2 = 2 * np.pi / V * (np.cross(a[2], a[0]))
    b3 = 2 * np.pi / V * (np.cross(a[0], a[1]))

    b1 = b1 / np.sqrt(np.inner(b1, b1))
    b2 = b2 / np.sqrt(np.inner(b2, b2))
    b3 = b3 / np.sqrt(np.inner(b3, b3))

    T = np.array([b1,b2,b3])
    invT = np.linalg.inv(T)

    vx = invT[0][0]*v1+invT[0][1]*v2+invT[0][2]*v3
    vy = invT[1][0]*v1+invT[1][1]*v2+invT[1][2]*v3
    vz = invT[2][0]*v1+invT[2][1]*v2+invT[2][2]*v3


    return vx, vy, vz

def create_hash_table(arrays):
    '''
    Create a hash table. Keys are the gvectors, values are the indices
    '''
    hash_table = {}
    for index, array in enumerate(arrays):
        hash_table[tuple(array)] = int(index)
    return hash_table

def get_index(hash_table, arrays):
    '''
    return the indices of the given arrays
    '''
    new_index = np.zeros(arrays.shape[0],dtype=int)
    for index, array in enumerate(arrays):
        new_index[index] = (hash_table.get(tuple(array),-1))
    return new_index

def wfn_derivative(main_class):
    dk = main_class.dk
    wfn_file_name = main_class.input_folder + 'wfn.h5'
    wfn_file = h5.File(wfn_file_name, 'r')
    a = wfn_file['/mf_header/crystal/alat'][()] * wfn_file['/mf_header/crystal/avec'][()] * 0.52917
    bdot = main_class.bdot
    nk = main_class.nk
    nb_all = main_class.nc_in_file+main_class.nv_in_file

    nk_sub = int(nk/7)
    dipole_matrix = np.zeros([nk_sub, nb_all, nb_all, 3], dtype=np.complex128)
    dipole_matrix_xyz = np.zeros([nk_sub, nb_all, nb_all, 3], dtype=np.complex128)

    invS_xp = np.zeros([nk, nb_all, nb_all], dtype=np.complex128)
    invS_xn = np.zeros([nk, nb_all, nb_all], dtype=np.complex128)
    invS_yp = np.zeros([nk, nb_all, nb_all], dtype=np.complex128)
    invS_yn = np.zeros([nk, nb_all, nb_all], dtype=np.complex128)
    invS_zp = np.zeros([nk, nb_all, nb_all], dtype=np.complex128)
    invS_zn = np.zeros([nk, nb_all, nb_all], dtype=np.complex128)

    gvecs = wfn_file['wfns/gvecs'][()]
    coeffs = wfn_file['wfns/coeffs'][:, :, :, 0] + 1j * wfn_file['wfns/coeffs'][:,:, :, 1]
    ngk = wfn_file['mf_header/kpoints/ngk'][()]
    k_index = np.hstack((np.array([0]), np.cumsum(ngk)))
    ng = wfn_file['/mf_header/gspace/ng'][()]

    for ik in range(nk_sub):
        gvecs_k = gvecs[k_index[ik]:k_index[ik + 1], :]
        coeffs_k = coeffs[:, :, k_index[ik]:k_index[ik + 1]]

        gvecs_k_xp = gvecs[k_index[ik+nk_sub*1]:k_index[ik + 1+nk_sub*1], :]
        coeffs_k_xp = coeffs[:, :, k_index[ik+nk_sub*1]:k_index[ik + 1+nk_sub*1]]

        gvecs_k_xn = gvecs[k_index[ik+nk_sub*2]:k_index[ik + 1+nk_sub*2], :]
        coeffs_k_xn = coeffs[:, :, k_index[ik+nk_sub*2]:k_index[ik + 1+nk_sub*2]]

        gvecs_k_yp = gvecs[k_index[ik+nk_sub*3]:k_index[ik + 1+nk_sub*3], :]
        coeffs_k_yp = coeffs[:, :, k_index[ik+nk_sub*3]:k_index[ik + 1+nk_sub*3]]

        gvecs_k_yn = gvecs[k_index[ik+nk_sub*4]:k_index[ik + 1+nk_sub*4], :]
        coeffs_k_yn = coeffs[:, :, k_index[ik+nk_sub*4]:k_index[ik + 1+nk_sub*4]]

        gvecs_k_zp = gvecs[k_index[ik+nk_sub*5]:k_index[ik + 1+nk_sub*5], :]
        coeffs_k_zp = coeffs[:, :, k_index[ik+nk_sub*5]:k_index[ik + 1+nk_sub*5]]

        gvecs_k_zn = gvecs[k_index[ik+nk_sub*6]:k_index[ik + 1+nk_sub*6], :]
        coeffs_k_zn = coeffs[:, :, k_index[ik+nk_sub*6]:k_index[ik + 1+nk_sub*6]]

        gspace_k_dic_xp = create_hash_table(gvecs_k_xp)
        new_index_k_xp = get_index(gspace_k_dic_xp, gvecs_k)

        coeffs_k_xp_new = coeffs_k_xp[:, :, new_index_k_xp]
        yesorno_xp = np.where(new_index_k_xp == -1)
        coeffs_k_xp_new[:, :, yesorno_xp] = 0

        gspace_k_dic_xn = create_hash_table(gvecs_k_xn)
        new_index_k_xn = get_index(gspace_k_dic_xn, gvecs_k)

        coeffs_k_xn_new = coeffs_k_xn[:, :, new_index_k_xn]
        yesorno_xn = np.where(new_index_k_xn == -1)
        coeffs_k_xn_new[:, :, yesorno_xn] = 0

        gspace_k_dic_yp = create_hash_table(gvecs_k_yp)
        new_index_k_yp = get_index(gspace_k_dic_yp, gvecs_k)

        coeffs_k_yp_new = coeffs_k_yp[:, :, new_index_k_yp]
        yesorno_yp = np.where(new_index_k_yp == -1)
        coeffs_k_yp_new[:, :, yesorno_yp] = 0

        gspace_k_dic_yn = create_hash_table(gvecs_k_yn)
        new_index_k_yn = get_index(gspace_k_dic_yn, gvecs_k)

        coeffs_k_yn_new = coeffs_k_yn[:, :, new_index_k_yn]
        yesorno_yn = np.where(new_index_k_yn == -1)
        coeffs_k_yn_new[:, :, yesorno_yn] = 0

        gspace_k_dic_zp = create_hash_table(gvecs_k_zp)
        new_index_k_zp = get_index(gspace_k_dic_zp, gvecs_k)

        coeffs_k_zp_new = coeffs_k_zp[:, :, new_index_k_zp]
        yesorno_zp = np.where(new_index_k_zp == -1)
        coeffs_k_zp_new[:, :, yesorno_zp] = 0

        gspace_k_dic_zn = create_hash_table(gvecs_k_zn)
        new_index_k_zn = get_index(gspace_k_dic_zn, gvecs_k)

        coeffs_k_zn_new = coeffs_k_zn[:, :, new_index_k_zn]
        yesorno_zn = np.where(new_index_k_zn == -1)
        coeffs_k_zn_new[:, :, yesorno_zn] = 0

        overlap_xp = np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 0, :]), coeffs_k_xp_new[:, 0, :]) \
                     + np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 1, :]), coeffs_k_xp_new[:, 1, :],
                                 optimize='optimal')

        overlap_xn = np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 0, :]), coeffs_k_xn_new[:, 0, :]) \
                     + np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 1, :]), coeffs_k_xn_new[:, 1, :],
                                 optimize='optimal')

        overlap_yp = np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 0, :]), coeffs_k_yp_new[:, 0, :]) \
                     + np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 1, :]), coeffs_k_yp_new[:, 1, :],
                                 optimize='optimal')

        overlap_yn = np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 0, :]), coeffs_k_yn_new[:, 0, :]) \
                     + np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 1, :]), coeffs_k_yn_new[:, 1, :],
                                 optimize='optimal')

        overlap_zp = np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 0, :]), coeffs_k_zp_new[:, 0, :]) \
                     + np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 1, :]), coeffs_k_zp_new[:, 1, :],
                                 optimize='optimal')

        overlap_zn = np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 0, :]), coeffs_k_zn_new[:, 0, :]) \
                     + np.einsum("mg,ng -> mn", np.conj(coeffs_k[:, 1, :]), coeffs_k_zn_new[:, 1, :],
                                 optimize='optimal')

        connected_overlap_xp = overlap_xp * (abs(overlap_xp) > 0.1)
        connected_overlap_xn = overlap_xn * (abs(overlap_xn) > 0.1)
        connected_overlap_yp = overlap_yp * (abs(overlap_yp) > 0.1)
        connected_overlap_yn = overlap_yn * (abs(overlap_yn) > 0.1)
        connected_overlap_zp = overlap_zp * (abs(overlap_zp) > 0.1)
        connected_overlap_zn = overlap_zn * (abs(overlap_zn) > 0.1)

        Uxp, Sxp, Vhxp = np.linalg.svd(connected_overlap_xp)
        Uxn, Sxn, Vhxn = np.linalg.svd(connected_overlap_xn)

        Uyp, Syp, Vhyp = np.linalg.svd(connected_overlap_yp)
        Uyn, Syn, Vhyn = np.linalg.svd(connected_overlap_yn)

        Uzp, Szp, Vhzp = np.linalg.svd(connected_overlap_zp)
        Uzn, Szn, Vhzn = np.linalg.svd(connected_overlap_zn)

        corrected_overlap_xp = overlap_xp @ (Vhxp.T.conj() @ Uxp.T.conj())
        corrected_overlap_xn = overlap_xn @ (Vhxn.T.conj() @ Uxn.T.conj())

        corrected_overlap_yp = overlap_yp @ (Vhyp.T.conj() @ Uyp.T.conj())
        corrected_overlap_yn = overlap_yn @ (Vhyn.T.conj() @ Uyn.T.conj())

        corrected_overlap_zp = overlap_zp @ (Vhzp.T.conj() @ Uzp.T.conj())
        corrected_overlap_zn = overlap_zn @ (Vhzn.T.conj() @ Uzn.T.conj())

        invS_xp[ik, :, :] = (Vhxp.T.conj() @ Uxp.T.conj())
        invS_xn[ik, :, :] = (Vhxn.T.conj() @ Uxn.T.conj())
        invS_yp[ik, :, :] = (Vhyp.T.conj() @ Uyp.T.conj())
        invS_yn[ik, :, :] = (Vhyn.T.conj() @ Uyn.T.conj())
        invS_zp[ik, :, :] = (Vhzp.T.conj() @ Uzp.T.conj())
        invS_zn[ik, :, :] = (Vhzn.T.conj() @ Uzn.T.conj())

        dipole_matrix[ik, :, :, 0] = 1j * (corrected_overlap_xp - corrected_overlap_xn) / np.sqrt(
            bdot[0, 0]) / dk / 2  # *2 shouldn't *2 if you want p instead of v
        dipole_matrix[ik, :, :, 1] = 1j * (corrected_overlap_yp - corrected_overlap_yn) / np.sqrt(
            bdot[1, 1]) / dk / 2  # *2 shouldn't *2 if you want p instead of v
        dipole_matrix[ik, :, :, 2] = 1j * (corrected_overlap_zp - corrected_overlap_zn) / np.sqrt(
            bdot[2, 2]) / dk / 2  # *2 shouldn't *2 if you want p instead of v
        print(str(ik) + "/" + str(int(nk/7)))


    wfn_file.close()
    dipole_matrix_xyz[:, :, :, 0], dipole_matrix_xyz[:, :, :, 1], dipole_matrix_xyz[:, :, :, 2] = b123_to_xyz(a,
                                                                                                              dipole_matrix[
                                                                                                              :, :, :,
                                                                                                              0],
                                                                                                              dipole_matrix[
                                                                                                              :, :, :,
                                                                                                              1],
                                                                                                              dipole_matrix[
                                                                                                              :, :, :,
                                                                                                              2])
    main_class.berry_phase = dipole_matrix_xyz
    main_class.invS_xp = invS_xp
    main_class.invS_xn = invS_xn
    main_class.invS_yp = invS_yp
    main_class.invS_yn = invS_yn
    main_class.invS_zp = invS_zp
    main_class.invS_zn = invS_zn
    # main_class.noeh_dipole_length = dipole_matrix_xyz[:, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc,
    #                          main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]
    return 0

def Ddipole(main_class):
    dk = main_class.dk
    wfn_file_name = main_class.input_folder + 'wfn.h5'
    wfn_file = h5.File(wfn_file_name, 'r')
    bdot = main_class.bdot
    nb_all = main_class.nc_in_file + main_class.nv_in_file
    a = wfn_file['/mf_header/crystal/alat'][()] * wfn_file['/mf_header/crystal/avec'][()] * 0.52917
    energy = wfn_file['mf_header/kpoints/el'][()]
    rk = wfn_file['mf_header/kpoints/rk'][()]

    nk_sub = int(main_class.nk / 7)

    xi_x = main_class.berry_phase[:, :, :, 0]
    xi_y = main_class.berry_phase[:, :, :, 1]
    xi_z = main_class.berry_phase[:, :, :, 2]

    invS_xp = main_class.invS_xp
    invS_xn = main_class.invS_xn
    invS_yp = main_class.invS_yp
    invS_yn = main_class.invS_yn
    invS_zp = main_class.invS_zp
    invS_zn = main_class.invS_zn

    input_file_dipole = main_class.input_folder + 'dipole_matrix.h5'
    f_dipole = h5.File(input_file_dipole, 'r')

    dipole_0 = f_dipole['dipole'][0*nk_sub:1*nk_sub, :, :, :]
    dipole_xp = f_dipole['dipole'][1 * nk_sub:2 * nk_sub, :, :, :]
    dipole_xn = f_dipole['dipole'][2 * nk_sub:3 * nk_sub, :, :, :]
    dipole_yp = f_dipole['dipole'][3 * nk_sub:4 * nk_sub, :, :, :]
    dipole_yn = f_dipole['dipole'][4 * nk_sub:5 * nk_sub, :, :, :]
    dipole_zp = f_dipole['dipole'][5 * nk_sub:6 * nk_sub, :, :, :]
    dipole_zn = f_dipole['dipole'][6 * nk_sub:7 * nk_sub, :, :, :]

    datax = dipole_0[:, :, :, 0]
    datay = dipole_0[:, :, :, 1]
    dataz = dipole_0[:, :, :, 2]

    L = np.zeros([nk_sub, nb_all, nb_all, 3], dtype=np.complex)

    totx = np.einsum('kam,kmb-> kab', xi_y, dataz) - np.einsum('kam,kmb-> kab', xi_z, datay)
    toty = np.einsum('kam,kmb-> kab', xi_z, datax) - np.einsum('kam,kmb-> kab', xi_x, dataz)
    totz = np.einsum('kam,kmb-> kab', xi_x, datay) - np.einsum('kam,kmb-> kab', xi_y, datax)

    energy = wfn_file['mf_header/kpoints/el'][0, 0:nk_sub, :]
    Ekm = np.einsum('km,v->kvm', energy, np.ones(nb_all))
    Ekv = np.einsum('kv,m->kvm', energy, np.ones(nb_all))
    energy_diff = (Ekm - Ekv)
    # degeneracy_remover = 0.00000001
    # with np.errstate(divide='ignore'):
    #     energy_diff_inverse = 1 / energy_diff
    #     energy_diff_inverse[abs(energy_diff) < degeneracy_remover] = 0
    # totx = np.einsum('kam,kam,kmb-> kab', datay, energy_diff_inverse,
    #                  dataz) - \
    #        np.einsum('kam,kam,kmb-> kab', dataz, energy_diff_inverse,
    #                  datay)
    # toty = np.einsum('kam,kam,kmb-> kab', dataz, energy_diff_inverse,
    #                  datax) - \
    #        np.einsum('kam,kam,kmb-> kab', datax, energy_diff_inverse,
    #                  dataz)
    # totz = np.einsum('kam,kam,kmb-> kab', datax, energy_diff_inverse,
    #                  datay) - \
    #        np.einsum('kam,kam,kmb-> kab', datay, energy_diff_inverse,
    #                  datax)

    # totx = np.einsum('kam,kmb,kmb-> kab', xi_y, energy_diff,xi_z) - np.einsum('kam,kmb,kmb-> kab', xi_z, energy_diff,xi_y)
    # toty = np.einsum('kam,kmb,kmb-> kab', xi_z, energy_diff,xi_x) - np.einsum('kam,kmb,kmb-> kab', xi_x, energy_diff,xi_z)
    # totz = np.einsum('kam,kmb,kmb-> kab', xi_x, energy_diff,xi_y) - np.einsum('kam,kmb,kmb-> kab', xi_y, energy_diff,xi_x)


    totx_intra = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)
    toty_intra = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)
    totz_intra = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)

    Ddipole_x_b1 = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)
    Ddipole_x_b2 = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)
    Ddipole_x_b3 = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)

    Ddipole_y_b1 = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)
    Ddipole_y_b2 = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)
    Ddipole_y_b3 = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)

    Ddipole_z_b1 = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)
    Ddipole_z_b2 = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)
    Ddipole_z_b3 = np.zeros([nk_sub, nb_all, nb_all], dtype=np.complex)

    for ik in range(nk_sub):
        Ddipole_x_b1[ik, :, :] = ((invS_xp[ik, :, :].T.conj()) @ dipole_xp[ik, :, :, 0] @ (invS_xp[ik, :, :])
                                  - (invS_xn[ik, :, :].T.conj()) @ dipole_xn[ik, :, :, 0] @ (
                                  invS_xn[ik, :, :])) / np.sqrt(bdot[0, 0]) / dk / 2

        Ddipole_x_b2[ik, :, :] = ((invS_yp[ik, :, :].T.conj()) @ dipole_yp[ik, :, :, 0] @ (invS_yp[ik, :, :])
                                  - (invS_yn[ik, :, :].T.conj()) @ dipole_yn[ik, :, :, 0] @ (
                                  invS_yn[ik, :, :])) / np.sqrt(bdot[1, 1]) / dk / 2

        Ddipole_x_b3[ik, :, :] = ((invS_zp[ik, :, :].T.conj()) @ dipole_zp[ik, :, :, 0] @ (invS_zp[ik, :, :])
                                  - (invS_zn[ik, :, :].T.conj()) @ dipole_zn[ik, :, :, 0] @ (
                                  invS_zn[ik, :, :])) / np.sqrt(bdot[2, 2]) / dk / 2

        Ddipole_y_b1[ik, :, :] = ((invS_xp[ik, :, :].T.conj()) @ dipole_xp[ik, :, :, 1] @ (invS_xp[ik, :, :])
                                  - (invS_xn[ik, :, :].T.conj()) @ dipole_xn[ik, :, :, 1] @ (
                                  invS_xn[ik, :, :])) / np.sqrt(bdot[0, 0]) / dk / 2

        Ddipole_y_b2[ik, :, :] = ((invS_yp[ik, :, :].T.conj()) @ dipole_yp[ik, :, :, 1] @ (invS_yp[ik, :, :])
                                  - (invS_yn[ik, :, :].T.conj()) @ dipole_yn[ik, :, :, 1] @ (
                                  invS_yn[ik, :, :])) / np.sqrt(bdot[1, 1]) / dk / 2

        Ddipole_y_b3[ik, :, :] = ((invS_zp[ik, :, :].T.conj()) @ dipole_zp[ik, :, :, 1] @ (invS_zp[ik, :, :])
                                  - (invS_zn[ik, :, :].T.conj()) @ dipole_zn[ik, :, :, 1] @ (
                                  invS_zn[ik, :, :])) / np.sqrt(bdot[2, 2]) / dk / 2

        Ddipole_z_b1[ik, :, :] = ((invS_xp[ik, :, :].T.conj()) @ dipole_xp[ik, :, :, 2] @ (invS_xp[ik, :, :])
                                  - (invS_xn[ik, :, :].T.conj()) @ dipole_xn[ik, :, :, 2] @ (
                                  invS_xn[ik, :, :])) / np.sqrt(bdot[0, 0]) / dk / 2

        Ddipole_z_b2[ik, :, :] = ((invS_yp[ik, :, :].T.conj()) @ dipole_yp[ik, :, :, 2] @ (invS_yp[ik, :, :])
                                  - (invS_yn[ik, :, :].T.conj()) @ dipole_yn[ik, :, :, 2] @ (
                                  invS_yn[ik, :, :])) / np.sqrt(bdot[1, 1]) / dk / 2

        Ddipole_z_b3[ik, :, :] = ((invS_zp[ik, :, :].T.conj()) @ dipole_zp[ik, :, :, 2] @ (invS_zp[ik, :, :])
                                  - (invS_zn[ik, :, :].T.conj()) @ dipole_zn[ik, :, :, 2] @ (
                                  invS_zn[ik, :, :])) / np.sqrt(bdot[2, 2]) / dk / 2

    Ddipole_x_x, Ddipole_x_y, Ddipole_x_z = b123_to_xyz(a, Ddipole_x_b1, Ddipole_x_b2, Ddipole_x_b3)
    Ddipole_y_x, Ddipole_y_y, Ddipole_y_z = b123_to_xyz(a, Ddipole_y_b1, Ddipole_y_b2, Ddipole_y_b3)
    Ddipole_z_x, Ddipole_z_y, Ddipole_z_z = b123_to_xyz(a, Ddipole_z_b1, Ddipole_z_b2, Ddipole_z_b3)

    totx_intra = 1j * Ddipole_z_y - 1j * Ddipole_y_z
    toty_intra = 1j * Ddipole_x_z - 1j * Ddipole_z_x
    totz_intra = 1j * Ddipole_y_x - 1j * Ddipole_x_y

    fact0 = 1
    fact = 1


    L[:, :, :, 0] = fact0*totx + totx_intra * fact
    L[:, :, :, 1] = fact0*toty + toty_intra * fact
    L[:, :, :, 2] = fact0*totz + totz_intra * fact

    main_class.L_length = L


    wfn_file.close()
    f_dipole.close()
    return 0

def Avck_derivative(main_class):
    dk = main_class.dk
    nv = main_class.nv
    nc = main_class.nc
    hovb = main_class.hovb
    wfn_file_name = main_class.input_folder + 'wfn.h5'
    wfn_file = h5.File(wfn_file_name, 'r')
    bdot = main_class.bdot
    nb_all = main_class.nc_in_file + main_class.nv_in_file
    a = wfn_file['/mf_header/crystal/alat'][()] * wfn_file['/mf_header/crystal/avec'][()] * 0.52917
    energy = wfn_file['mf_header/kpoints/el'][()]
    rk = wfn_file['mf_header/kpoints/rk'][()]

    nk_sub = int(main_class.nk / 7)

    input_file_dipole = main_class.input_folder + 'dipole_matrix.h5'
    f_dipole = h5.File(input_file_dipole, 'r')

    dipole_0 = f_dipole['dipole'][0*nk_sub:1*nk_sub, :, :, :]
    dipole_xp = f_dipole['dipole'][1 * nk_sub:2 * nk_sub, :, :, :]
    dipole_xn = f_dipole['dipole'][2 * nk_sub:3 * nk_sub, :, :, :]
    dipole_yp = f_dipole['dipole'][3 * nk_sub:4 * nk_sub, :, :, :]
    dipole_yn = f_dipole['dipole'][4 * nk_sub:5 * nk_sub, :, :, :]
    dipole_zp = f_dipole['dipole'][5 * nk_sub:6 * nk_sub, :, :, :]
    dipole_zn = f_dipole['dipole'][6 * nk_sub:7 * nk_sub, :, :, :]

    invS_xp = main_class.invS_xp
    invS_xn = main_class.invS_xn
    invS_yp = main_class.invS_yp
    invS_yn = main_class.invS_yn
    invS_zp = main_class.invS_zp
    invS_zn = main_class.invS_zn

    invS_xp_v = invS_xp[:, hovb-nv:hovb, hovb-nv:hovb]
    invS_xn_v = invS_xn[:, hovb - nv:hovb, hovb - nv:hovb]
    invS_yp_v = invS_yp[:, hovb-nv:hovb, hovb-nv:hovb]
    invS_yn_v = invS_yn[:, hovb - nv:hovb, hovb - nv:hovb]
    invS_zp_v = invS_zp[:, hovb-nv:hovb, hovb-nv:hovb]
    invS_zn_v = invS_zn[:, hovb - nv:hovb, hovb - nv:hovb]

    invS_xp_c = invS_xp[:, hovb:hovb+nc, hovb:hovb+nc]
    invS_xn_c = invS_xn[:, hovb:hovb + nc, hovb:hovb + nc]
    invS_yp_c = invS_yp[:, hovb:hovb + nc, hovb:hovb + nc]
    invS_yn_c = invS_yn[:, hovb:hovb + nc, hovb:hovb + nc]
    invS_zp_c = invS_zp[:, hovb:hovb + nc, hovb:hovb + nc]
    invS_zn_c = invS_zn[:, hovb:hovb + nc, hovb:hovb + nc]


    avck_0 = main_class.avck[0*nk_sub:1*nk_sub,:,:,:]   #kvcs
    avck_xp = main_class.avck[1 * nk_sub:2 * nk_sub, :, :, :]
    avck_xn = main_class.avck[2 * nk_sub:3 * nk_sub, :, :, :]
    avck_yp = main_class.avck[3 * nk_sub:4 * nk_sub, :, :, :]
    avck_yn = main_class.avck[4 * nk_sub:5 * nk_sub, :, :, :]
    avck_zp = main_class.avck[5 * nk_sub:6 * nk_sub, :, :, :]
    avck_zn = main_class.avck[6 * nk_sub:7 * nk_sub, :, :, :]

    Davck_x = (avck_xp-avck_xn) / np.sqrt(bdot[0, 0]) / dk / 2
    Davck_y = (avck_yp - avck_yn) / np.sqrt(bdot[1, 1]) / dk / 2
    Davck_z = (avck_zp - avck_zn) / np.sqrt(bdot[2, 2]) / dk / 2

    Davck_x_2 = np.zeros_like(Davck_x)
    idx = list(range(main_class.nv - 1, -1, -1))
    inds = np.ix_(range(nk_sub), idx, range(main_class.nc), range(main_class.nxct))
    avck_xp = avck_xp[inds]
    avck_xn = avck_xn[inds]
    test01 = np.zeros([nk_sub, 4, 8], dtype=np.complex)
    test02 = np.zeros([nk_sub, 4, 8], dtype=np.complex)
    test1 = np.zeros_like(Davck_x)
    test2 = np.zeros_like(Davck_x)

    Ddipole_x_b1 = np.zeros([nk_sub, 4, 8], dtype=np.complex)


    for ik in range(nk_sub):
        Ddipole_x_b1[ik, :, :] = ((invS_xp_v[ik, :, :].T.conj()) @ dipole_xp[ik, hovb - nv:hovb, hovb:hovb + nc, 0] @ (invS_xp_c[ik, :, :])
                                  - (invS_xn_v[ik, :, :].T.conj()) @ dipole_xn[ik, hovb - nv:hovb, hovb:hovb + nc, 0] @ (
                                  invS_xn_c[ik, :, :])) / np.sqrt(bdot[0, 0]) / dk / 2
        test01[ik, :, :] = (invS_xp_v[ik, :, :].T.conj()) @ dipole_xp[ik, hovb - nv:hovb, hovb:hovb + nc, 0] @ (invS_xp_c[ik, :, :])
        test02[ik, :, :] = invS_xn_v[ik, :, :].T.conj() @ dipole_xn[ik, hovb - nv:hovb, hovb:hovb + nc, 0] @ (invS_xn_c[ik, :, :])



    for ist in range(main_class.nxct):
        for ik in range(nk_sub):
            Davck_x_2[ik, :, :, ist] = (invS_xp_v[ik, :, :].T.conj() @ (avck_xp[ik, :, :, ist]) @ (
                invS_xp_c[ik, :, :]) - invS_xn_v[ik, :, :].T.conj() @ (avck_xn[ik, :, :, ist]) @ (
                                           invS_xn_c[ik, :, :])) / np.sqrt(bdot[0, 0]) / dk / 2
            test1[ik,:,:,ist] = (invS_xp_v[ik, :, :]) @ (avck_xp[ik, :, :, ist]) @ (invS_xp_c[ik, :, :].T.conj())
            test2[ik, :, :, ist] = (invS_xn_v[ik, :, :]) @ (avck_xn[ik, :, :, ist]) @ (invS_xn_c[ik, :, :].T.conj())
    test11 = np.einsum('kvcs,kvc->ks', avck_xp, dipole_xp[:, hovb - nv:hovb, hovb:hovb + nc, 0])
    test22 = np.einsum('kvcs,kvc->ks', avck_xn, dipole_xn[:, hovb - nv:hovb, hovb:hovb + nc, 0])

    new1 = np.einsum('kvcs,kvc->ks',test1, test01)
    new2 = np.einsum('kvcs,kvc->ks',test2, test02)

    result11 = np.sum(test11[:,:])
    result22 = np.sum(test22[:,:])

    result11_new = np.sum(new1)
    result22_new = np.sum(new2)

    #print((invS_xp_v[0, :, :].T.conj())@invS_xp_v[0, :, :])




    print('test')
    wfn_file.close()
    f_dipole.close()

def Avck_derivative2(main_class):
    dk = main_class.dk
    nv = main_class.nv
    nc = main_class.nc
    hovb = main_class.hovb
    wfn_file_name = main_class.input_folder + 'wfn.h5'
    wfn_file = h5.File(wfn_file_name, 'r')
    bdot = main_class.bdot
    nb_all = main_class.nc_in_file + main_class.nv_in_file
    a = wfn_file['/mf_header/crystal/alat'][()] * wfn_file['/mf_header/crystal/avec'][()] * 0.52917
    energy = wfn_file['mf_header/kpoints/el'][()]
    rk = wfn_file['mf_header/kpoints/rk'][()]

    nk_sub = int(main_class.nk / 7)

    input_file_dipole = main_class.input_folder + 'dipole_matrix.h5'
    f_dipole = h5.File(input_file_dipole, 'r')

    dipole_0 = f_dipole['dipole'][0*nk_sub:1*nk_sub, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]
    dipole_xp = f_dipole['dipole'][1 * nk_sub:2 * nk_sub, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]
    dipole_xn = f_dipole['dipole'][2 * nk_sub:3 * nk_sub, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]
    dipole_yp = f_dipole['dipole'][3 * nk_sub:4 * nk_sub, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]
    dipole_yn = f_dipole['dipole'][4 * nk_sub:5 * nk_sub, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]
    dipole_zp = f_dipole['dipole'][5 * nk_sub:6 * nk_sub, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]
    dipole_zn = f_dipole['dipole'][6 * nk_sub:7 * nk_sub, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]



    energy = main_class.energy_dft
    newE_0 = np.zeros_like(dipole_0)
    newE_xp = np.zeros_like(dipole_0)
    newE_xn = np.zeros_like(dipole_0)
    newE_yp = np.zeros_like(dipole_0)
    newE_yn = np.zeros_like(dipole_0)
    newE_zp = np.zeros_like(dipole_0)
    newE_zn = np.zeros_like(dipole_0)

    for ik in range(nk_sub):
        for ib1 in range(main_class.nv + main_class.nc):
            for ib2 in range(main_class.nv + main_class.nc):
                energy_diff_for_cancel_diple = energy[ik, ib1] - energy[
                    ik, ib2]
                if np.abs(energy_diff_for_cancel_diple) > 0.0000001:
                    energy_diff_for_cancel_diple_inv = 1 / energy_diff_for_cancel_diple
                else:
                    energy_diff_for_cancel_diple_inv = 0

                newE_0[ik, ib1, ib2, :] = dipole_0[ik, ib1, ib2, :] * energy_diff_for_cancel_diple_inv
                newE_xp[ik, ib1, ib2, :] = dipole_xp[ik, ib1, ib2, :] * energy_diff_for_cancel_diple_inv
                newE_xn[ik, ib1, ib2, :] = dipole_xn[ik, ib1, ib2, :] * energy_diff_for_cancel_diple_inv
                newE_yp[ik, ib1, ib2, :] = dipole_yp[ik, ib1, ib2, :] * energy_diff_for_cancel_diple_inv
                newE_yn[ik, ib1, ib2, :] = dipole_yn[ik, ib1, ib2, :] * energy_diff_for_cancel_diple_inv
                newE_zp[ik, ib1, ib2, :] = dipole_zp[ik, ib1, ib2, :] * energy_diff_for_cancel_diple_inv
                newE_zn[ik, ib1, ib2, :] = dipole_zn[ik, ib1, ib2, :] * energy_diff_for_cancel_diple_inv

    dipole_0_2 = newE_0[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]
    dipole_xp_2 = newE_xp[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]
    dipole_xn_2 = newE_xn[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]
    dipole_yp_2 = newE_yp[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]
    dipole_yn_2 = newE_yn[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]
    dipole_zp_2 = newE_zp[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]
    dipole_zn_2 = newE_zn[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]

    avck_0 = main_class.avck[0*nk_sub:1*nk_sub,:,:,:]   #kvcs
    avck_xp = main_class.avck[1 * nk_sub:2 * nk_sub, :, :, :]
    avck_xn = main_class.avck[2 * nk_sub:3 * nk_sub, :, :, :]
    avck_yp = main_class.avck[3 * nk_sub:4 * nk_sub, :, :, :]
    avck_yn = main_class.avck[4 * nk_sub:5 * nk_sub, :, :, :]
    avck_zp = main_class.avck[5 * nk_sub:6 * nk_sub, :, :, :]
    avck_zn = main_class.avck[6 * nk_sub:7 * nk_sub, :, :, :]

    # dipole_0_2 =np.zeros_like(dipole_0)
    # dipole_xp_2 = np.zeros_like(dipole_xp)
    # dipole_xn_2 = np.zeros_like(dipole_xn)
    # dipole_yp_2 = np.zeros_like(dipole_yp)
    # dipole_yn_2 = np.zeros_like(dipole_yn)
    # dipole_zp_2 = np.zeros_like(dipole_zp)
    # dipole_zn_2 = np.zeros_like(dipole_zn)
    #
    #
    # energy = main_class.energy_dft
    #
    # for ik in range(nk_sub):
    #     for ib1 in range(main_class.nv + main_class.nc):
    #         for ib2 in range(main_class.nv + main_class.nc):
    #             energy_diff_for_cancel_diple = energy[ik, ib1] - energy[
    #                 ik, ib2]
    #             if np.abs(energy_diff_for_cancel_diple) > 0.0000001:
    #                 energy_diff_for_cancel_diple_inv = 1 / energy_diff_for_cancel_diple
    #             else:
    #                 energy_diff_for_cancel_diple_inv = 0
    #
    #             newE[ik, ib1, ib2, :] = E[ik, ib1, ib2, :] * energy_diff_for_cancel_diple_inv
    # main_class.E_kvc = newE[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :]

    idx = list(range(main_class.nv - 1, -1, -1))
    inds = np.ix_(range(nk_sub), idx, range(main_class.nc), range(3))
    Ekvc_0 = dipole_0_2[inds]
    Ekvc_xp = dipole_xp_2[inds]
    Ekvc_xn = dipole_xn_2[inds]
    Ekvc_yp = dipole_yp_2[inds]
    Ekvc_yn = dipole_yn_2[inds]
    Ekvc_zp = dipole_zp_2[inds]
    Ekvc_zn = dipole_zn_2[inds]


    ME_0 = np.einsum('kvcs,kvcd->sd', avck_0, Ekvc_0)
    ME_xp = np.einsum('kvcs,kvcd->sd', avck_xp, Ekvc_xp)
    ME_xn = np.einsum('kvcs,kvcd->sd', avck_xn, Ekvc_xn)
    ME_yp = np.einsum('kvcs,kvcd->sd', avck_yp, Ekvc_yp)
    ME_yn = np.einsum('kvcs,kvcd->sd', avck_yn, Ekvc_yn)
    ME_zp = np.einsum('kvcs,kvcd->sd', avck_zp, Ekvc_zp)
    ME_zn = np.einsum('kvcs,kvcd->sd', avck_zn, Ekvc_zn)

    L_correction = np.zeros([main_class.nxct,3], dtype=np.complex128)

    L_correction[:,0] = 1j * (ME_yp[:,2]-ME_yn[:,2])/ np.sqrt(bdot[1, 1]) / dk / 2 - 1j * (ME_zp[:,1]-ME_zn[:,1])/ np.sqrt(bdot[2, 2]) / dk / 2
    L_correction[:,1] = 1j * (ME_zp[:, 0] - ME_zn[:, 0]) / np.sqrt(bdot[2, 2]) / dk / 2 - 1j * (ME_xp[:, 2] - ME_xn[:, 2]) / np.sqrt(bdot[0, 0]) / dk / 2
    L_correction[:,2] = 1j * (ME_xp[:, 1] - ME_xn[:, 1]) / np.sqrt(bdot[0, 0]) / dk / 2 - 1j * (ME_yp[:, 0] - ME_yn[:, 0]) / np.sqrt(bdot[1, 1]) / dk / 2

    main_class.L_correction = L_correction
    print('test')


