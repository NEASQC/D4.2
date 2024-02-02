#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(1,'D4.2/lib')
from lib.Github_calc_Hamilt import save_H_into_dict
from lib.Github_calc_Energy import save_E_into_dict

from pyscf import gto, scf
import numpy as np

def test_H2():
    """
    This functions shows how to obtain H2 ground state energy curve with qUCC and HE method.
    """

    l_R = list(np.linspace(0.2,2,10)) + [5]

    for R_H2 in l_R:

        mol = gto.Mole()
        mol.atom = f'H 0 0 0; H 0 0 {R_H2}'
        mol.basis = 'sto-3g'
        mol.spin = 0
        mol.build(0,0)

        m_mol = scf.RHF(mol)
        m_mol.kernel()

        #############################################

        save_filename = 'test_github_neasqc.H2'
        nb_lumo = 1
        nb_homo = 1
        dic_H_save = save_H_into_dict(R_H2, save_filename, mol, m_mol, nb_homo, nb_lumo)

        #############################################

        hamilt_filename = 'test_github_neasqc.H2.H.pickle'
        save_filename = 'test_github_neasqc.H2'
        

        ### qUCC calculation 
        dic_E_save = save_E_into_dict(R_H2, hamilt_filename, save_filename, mol, m_mol, nb_homo, nb_lumo, ansatz="qUCC", nbshots=0, N_trials=1)

        ### HE calculation
        dic_E_save = save_E_into_dict(R_H2, hamilt_filename, save_filename, mol, m_mol, nb_homo, nb_lumo, ansatz="HE", nbshots=0, d=1, N_trials=1)


    return dic_E_save




