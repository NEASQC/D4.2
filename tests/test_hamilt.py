#!/usr/bin/env python
# coding: utf-8


import numpy as np

from .Github_calc_Hamilt import ob_tb_integ

from pyscf import gto, scf

def test_ob_tb_integ(R_H2= 0.74):
    """
    Input :
        - R_H2 : bond length of H2
    Output :
        - 1body and 2body integrals of H2 molecule
    """
    mol = gto.Mole()
    mol.atom = f'H 0 0 0; H 0 0 {R_H2}'
    mol.basis = 'sto-3g'
    mol.spin = 0
    mol.build(0,0)

    m_mol = scf.RHF(mol)
    m_mol.kernel()

    one_body_integ, two_body_integ = ob_tb_integ(mol, m_mol)
    return one_body_integ, two_body_integ


from .Github_calc_Hamilt import H_with_active_space_reduction

def test_H_with_active_space_reduction(R_OH=0.96):
    """
    Input :
        - R_OH : distance between H and O atom in H2O (angles are fixed)
    Returns, for H2O molecule with R_OH:
        - H_active : active space Hamiltonian (with 1 HOMO and 1 LUMO)
        - active_inds : list of index of active orbitals
        - occ_inds : list of index of occupied orbitals
        - noons : list of NOONs
        - orbital_energies : list of orbital_energies
        - nels : number of electrons
        - eps1, eps2 : thresholds to proceed the active space reduction
    """
    mol = gto.Mole()
    mol.verbose = 0
    H10 = f'H {R_OH*np.cos(np.radians(104.45))} {R_OH*np.sin(np.radians(104.45))} 0; '
    H10 += f'H {R_OH} 0 0; '
    H10 += f'O 0 0 0'
    mol.atom = H10
    mol.basis = 'sto-3g'
    mol.spin = 0
    mol.build(0,0)
    m_mol = scf.RHF(mol)
    m_mol.kernel()
    one_body_integ, two_body_integ = ob_tb_integ(mol, m_mol)

    nb_homo, nb_lumo = 1, 1
    H_active, active_inds, occ_inds, noons, orbital_energies, nels, eps1, eps2 = H_with_active_space_reduction(one_body_integ, two_body_integ, mol, m_mol, nb_homo, nb_lumo)
    
    return H_active, active_inds, occ_inds, noons, orbital_energies, nels, eps1, eps2

