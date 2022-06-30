#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pickle
import qat.dqs.qchem.pyscf_tools as pyscf
from pyscf import gto, scf, ao2mo, ci
from functools import reduce
from qat.dqs.qchem import transform_integrals_to_new_basis
from qat.dqs.qchem.ucc import get_active_space_hamiltonian
import scipy


def ob_tb_integ(mol,m_mol):
    """
    Input : PySCF molecule
    Output : one-body & two-body integrals
    """
    
    ########
    # 1 : calculation of one_body_integral
    ########
    n_orbitals = m_mol.mo_coeff.shape[1]
    one_body_compressed = reduce(np.dot, (m_mol.mo_coeff.T, m_mol.get_hcore(), m_mol.mo_coeff))
    one_body_integ = one_body_compressed.reshape(n_orbitals, n_orbitals).astype(float)

    ########
    # 2 : calculation of two_body_integral
    ########
    
    two_body_compressed = ao2mo.kernel(mol, m_mol.mo_coeff)
    two_body_integ = ao2mo.restore(1, two_body_compressed, n_orbitals)
    two_body_integ = np.asarray(two_body_integ.transpose(0, 2, 3, 1), order='C')

    return one_body_integ, two_body_integ

def H_with_active_space_reduction(one_body_integ, two_body_integ, mol, m_mol, nb_homo, nb_lumo):
    """
    Input : one-body & two-body integrals, PySCF molecule, number of HOMO and LUMO 
    Output :
        - H_active : Hamiltonian after active space reduction (orbital freezing)
        - active_inds, occ_inds :  list of index of active/occupied orbitals
        - noons : list of natural-orbital occupation numbers 
        - orbital energies : list of energies of each molecular orbital
        - nels : total number of electrons
    """
    
    ########
    # 1 : preparation
    ########
    
    nels = mol.nelectron
    nuclear_repulsion = mol.energy_nuc()
    orbital_energies = m_mol.mo_energy
    
    ci_mol = ci.CISD(m_mol.run()).run()
    rdm1 = ci_mol.make_rdm1()
    noons, basis_change = np.linalg.eigh(rdm1)
    noons = list(reversed(noons))
    basis_change = np.flip(basis_change, axis=1)
    one_body_integrals, two_body_integrals = transform_integrals_to_new_basis(one_body_integ,
                                                                              two_body_integ,
                                                                              basis_change,
                                                                              old_version=False)
    
    
    ########
    # 2 : calculation of the reduced Hamiltonian
    ########
    assert nb_homo >= 0, 'nb_homo >= 0'
    assert nb_lumo >= 0, 'nb_homo >= 0'
    assert nb_homo <= nels//2, f'nb_homo <= {nels//2}'
    assert nb_lumo <= len(noons)-nels//2, f'nb_lumo <= {len(noons)-nels//2}'

    homo_min = nels//2-nb_homo
    lumo_max = nels//2 + nb_lumo
    if homo_min == 0:
        eps1 = 0
    else:
        eps1 = 2 - (noons[homo_min-1]+noons[homo_min])/2
    eps2 = noons[lumo_max-1]

    H_active, active_inds, occ_inds = get_active_space_hamiltonian(one_body_integrals,
                                                  two_body_integrals, 
                                                  noons, nels, nuclear_repulsion, threshold_1 = eps1, threshold_2 = eps2)
    
    return H_active, active_inds, occ_inds, noons, orbital_energies, nels, eps1, eps2

########################
### Hamiltonian save ###
########################

def save_H_into_dict(l1, save_filename, mol, m_mol, nb_homo, nb_lumo, calc_E_exact = False):
    """
    Input : l1 (varying parameter), filename for saving, PySCF molecule, number of HOMO and LUMO, choice of exact energy calculation
    Output :
        - dic_H_save : dictionnary contained in save_filename.pickle with
               * 1st key : varying parameter (e.g. : bond length of a molecule)
               * 2nd key : basis set
               * 3rd key : nb_homo (characterizes the active space reduction)
               * 4th key : nb_lumo (characterizes the active space reduction)
               * Then :
                    - H_active : Hamiltonian after active space reduction (orbital freezing)
                    - active_inds, occ_inds :  list of index of active/occupied orbitals
                    - noons : list of natural-orbital occupation numbers 
                    - orbital energies : list of energies of each molecular orbital
                    - nels : total number of electrons
                    - E_exact : exact ground state energy of the system, obtained with diagonalization
    """
    try:
        with open(f'{save_filename}.H.pickle','rb') as f1:
            dic_H_save = pickle.load(f1)
    except:
        print(f'Error : The dictionary {save_filename}.pickle doesn\'t exist.\n => Creation of a new one')
        dic_H_save = {}
    try:
        dic_H_save[str(l1)]
    except:
        dic_H_save[str(l1)] = {}
        
    ob, tb = ob_tb_integ(mol,m_mol)
    print(f'Nb of qubits (before reduction) : {2*ob.shape[0]}')
    
    H_active, active_inds, occ_inds, noons, orbital_energies, nels, eps1, eps2 = H_with_active_space_reduction(ob, tb, mol, m_mol, nb_homo, nb_lumo)
    
    print(f'··· nb_homo = {nb_homo} | nb_lumo = {nb_lumo}')
    print(f'··· noons = {noons}')
    print(f'··· occ_inds = {occ_inds}')
    print(f'··· active_inds = {active_inds}')

    print(f'Nb of qubits (after reduction) : {H_active.nbqbits}')
    
    try:
        dic_H_save[str(l1)][mol.basis]
    except:
        dic_H_save[str(l1)][mol.basis] = {}
    try:
        dic_H_save[str(l1)][mol.basis][str(nb_homo)]
    except:
        dic_H_save[str(l1)][mol.basis][str(nb_homo)] = {}
    try:
        dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]
    except:
        dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)] = {}
        
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['H_active'] = H_active
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['active_inds'] = active_inds
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['occ_inds'] = occ_inds
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['noons'] = noons
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['orbital_energies'] = orbital_energies
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['nels'] = nels
    if calc_E_exact == True:
        EIGVAL, _ = scipy.sparse.linalg.eigs(H_active.get_matrix(sparse=True))
        E_exact = np.min(np.real(EIGVAL))
        dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['E_exact'] = E_exact
    else:
        dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['E_exact'] = None
        
    with open(f'{save_filename}.H.pickle','wb') as f2:
        pickle.dump(dic_H_save,f2)
        print(f'=> The dictionary of results is saved in {save_filename}.H.pickle')

    return dic_H_save

##############################
### Distortions of benzene ###
##############################

def build_benz_dist_1(alpha, basis='sto-3g'):
    """
    Input : alpha (varying parameter), basis set
    Ouput : benzene PySCF molecule under distortion 1 (according to paper ...)
    """
    mol_benz = gto.Mole()
    mol_benz.verbose = 5
    mol_benz.output = 'benz.log'
    R_CC = 1.39*alpha
    R_CH = 1.09
    c_h = ''
    for i in range(6):
        angle = np.pi/6 + i*np.pi/3
        x, y = R_CC*np.cos(angle), R_CC*np.sin(angle)
        x_H, y_H = (R_CC+R_CH)*np.cos(angle), (R_CC+R_CH)*np.sin(angle)
        c_h += f'C {x} {y} 0; H {x_H} {y_H} 0;'
    mol_benz.atom = c_h
    mol_benz.basis = basis
    mol_benz.spin = 0
    mol_benz.build(0,0)
    m_benz = scf.RHF(mol_benz)
    m_benz.kernel()
    return mol_benz, m_benz

def build_benz_dist_2(alpha, basis='sto-3g'):
    """
    Input : alpha (varying parameter), basis set
    Ouput : benzene PySCF molecule under distortion 2 (according to paper ...)
    """
    mol_benz = gto.Mole()
    mol_benz.verbose = 5
    mol_benz.output = 'benz.log'
    R1 = 1.39
    R_CH = 1.09
    R2 = 2*R1*np.cos(np.pi/6)*alpha
    x1 = R1*np.sin(np.pi/6)
    X = [0,R1,R1+x1,R1,0,-x1]
    Y = [0,0,R2/2,R2,R2,R2/2]
    X.append(X[0])
    Y.append(Y[0])
    X_H, Y_H = [], []
    xh = R_CH*np.cos(np.pi/3)
    yh = R_CH*np.sin(np.pi/3)
    X_H = [-xh,xh,R_CH,xh,-xh,-R_CH]
    Y_H = [-yh,-yh,0,yh,yh,0]
    c_h = ''
    for i in range(6):
        X_H[i] += X[i]
        Y_H[i] += Y[i]
        c_h += f'C {X[i]} {Y[i]} 0; H {X_H[i]} {Y_H[i]} 0;'
    mol_benz.atom = c_h
    mol_benz.basis = basis
    mol_benz.spin = 0
    mol_benz.build(0,0)
    m_benz = scf.RHF(mol_benz)
    m_benz.kernel()
    return mol_benz, m_benz

def build_benz_dist_3(alpha, basis='sto-3g'):
    """
    Input : alpha (varying parameter), basis set
    Ouput : benzene PySCF molecule under distortion 3 (according to paper ...)
    """
    mol_benz = gto.Mole()
    mol_benz.verbose = 5
    mol_benz.output = 'benz.log'
    R10 = 1.39
    R_CH = 1.09
    R2 = 2*R10*np.cos(np.pi/6)
    x1 = R10*np.sin(np.pi/6)
    R1 = R10*alpha
    X = [0,R1,R1+x1,R1,0,-x1]
    Y = [0,0,R2/2,R2,R2,R2/2]
    X.append(X[0])
    Y.append(Y[0])
    X_H, Y_H = [], []
    xh = R_CH*np.cos(np.pi/3)
    yh = R_CH*np.sin(np.pi/3)
    X_H = [-xh,xh,R_CH,xh,-xh,-R_CH]
    Y_H = [-yh,-yh,0,yh,yh,0]
    c_h = ''
    for i in range(6):
        X_H[i] += X[i]
        Y_H[i] += Y[i]
        c_h += f'C {X[i]} {Y[i]} 0; H {X_H[i]} {Y_H[i]} 0;'
    mol_benz.atom = c_h
    mol_benz.basis = basis
    mol_benz.spin = 0
    mol_benz.build(0,0)
    m_benz = scf.RHF(mol_benz)
    m_benz.kernel()
    return mol_benz, m_benz


def full_hamilt_computation(dist, alpha, basis, nb_homo, nb_lumo, calc_E_exact = False):
    """
    Input : 
        - dist : choice of distortion applied to benzene
        - alpha : distorsion parameter
        - basis : basis set
        - nb_homo, nb_lumo : characterizes the active space reduction
        - calc_E_exact : choice of exact energy calculation
    Output : 
        - dictionary containing Hamiltonian of this benzene created by save_H_into_dict 
    """
    if dist==1:
        mol, m_mol = build_benz_dist_1(alpha, basis)
    elif dist==2:
        mol, m_mol = build_benz_dist_2(alpha, basis)
    elif dist==3:
        mol, m_mol = build_benz_dist_3(alpha, basis)
    else:
        print(f'Error : dist = 1, 2 or 3')
    
    save_filename = f'benzene_dist{dist}'
    dic_H_save = save_H_into_dict(alpha, save_filename, mol, m_mol, nb_homo, nb_lumo, calc_E_exact)
    return dic_H_save
    
    
    

