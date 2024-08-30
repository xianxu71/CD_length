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
    hovb = main_class.hovb
    nv = main_class.nv
    nc = main_class.nc
    dk = main_class.dk
    wfn_file_name = main_class.input_folder + 'wfn.h5'
    wfn_file = h5.File(wfn_file_name, 'r')
    a = main_class.a
    bdot = main_class.bdot
    nk = main_class.nk
    nb_all = main_class.nc_in_file+main_class.nv_in_file
    energy = wfn_file['mf_header/kpoints/el']
    rk = wfn_file['mf_header/kpoints/rk'][()]

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
    main_class.noeh_dipole_length = dipole_matrix_xyz[:, main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc,
                             main_class.nv_in_file - main_class.nv:main_class.nv_in_file + main_class.nc, :]
    return 0