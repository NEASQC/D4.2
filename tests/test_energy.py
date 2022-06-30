#!/usr/bin/env python
# coding: utf-8

import numpy as np

#################
### HE Method ###
#################

from qat.lang.AQASM import Program
from qat.interop.qiskit import qlm_to_qiskit

from .Github_calc_Energy import HE_circuit_for_ansatz

def test_print_circuit_HE(nbqbits, depth):
    """
    Input : 
        - nbqbits : number of qubits
        - depth : depth of the ansatz
    Output :
        - qiskit_circuit : HE circuit in qiskit format (only to be printed)
    
    """
    depth = 1
    nbqbits = 4
    nb_parameters = depth*2*(1+3*(nbqbits-1))
    prog = Program()
    reg = prog.qalloc(nbqbits)
    theta = [prog.new_var(float, '\u03B8'+'%s'%i) for i in range(nb_parameters)]
    theta = np.asarray(theta)    
    prog.apply(HE_circuit_for_ansatz(theta, nbqbits), reg)
    circ = prog.to_circ()
    qiskit_circuit = qlm_to_qiskit(circ)
    return qiskit_circuit.draw(output='mpl')



from qat.dqs import FermionHamiltonian
from qat.core import Term
from qat.dqs.transforms import transform_to_jw_basis
from qat.qpus import LinAlg

from .Github_calc_Energy import fun_HE_ansatz

def test_fun_HE_ansatz(nbqbits, nbshots):
    """
    Input :
        - nbqbits : number of qubits 
        - nbshots : number of shots for quantum measurement
    Output :
        - val : expectation value of H = 1*a^\dagger_0*a_0
                with |psi> containing each nqbits-state with uniform probability.
                (val must be equal to 0.5 with nbshots = 0)
    """
    Hamilt_test = FermionHamiltonian(nbqbits, [Term(1, "Cc", [0, 0])])
    Hamilt_test_sp = transform_to_jw_basis(Hamilt_test)
    theta = np.zeros(depth*2*(1+3*(nbqbits-1)))
    qpu = LinAlg()

    global compteur
    compteur = 0
    val = fun_HE_ansatz(Hamilt_test_sp, theta, nbshots, qpu)
    return val


###################
### qUCC Method ###
###################

from .Github_calc_Energy import ucc_ansatz_calc

from qat.dqs import ElectronicStructureHamiltonian

def test_print_circuit_quccsd(nbqbits):
    """
    Input : 
        - nbqbits : number of qubits
    Output :
        - qiskit_circuit : qUCCSD circuit in qiskit format (only to be printed)
    """
    hpq = np.zeros((nbqbits,nbqbits))
    hpq[0,0] = 1
    Hamilt_test = ElectronicStructureHamiltonian(hpq)
    active_inds = [_ for _ in range(nbqbits//2)]
    occ_inds = []
    noons = np.random.random(nbqbits//2)
    orbital_energies = np.random.random(nbqbits//2)
    nels = nbqbits//2
    Hamilt_test_sp, qprog, theta0 = ucc_ansatz_calc(Hamilt_test, active_inds, occ_inds, noons, orbital_energies, nels)
    prog = Program()
    reg = prog.qalloc(nbqbits)
    theta = [prog.new_var(float, '\u03B8'+'%s'%i) for i in range(len(theta0))]
    prog.apply(qprog(theta), reg)
    circ = prog.to_circ()
    qiskit_circuit = qlm_to_qiskit(circ)
    return qiskit_circuit.draw(output='mpl')
    

from .Github_calc_Energy import fun_qucc_ansatz

def test_fun_qucc_ansatz(nbqbits, nbshots):
    """
    Input :
        - nbqbits : number of qubits 
        - nbshots : number of shots for quantum measurement
    Output :
        - val : expectation value of H = 1*a^\dagger_0*a_0
                with |psi> containing qUCCSD ansatz.
                (val must be equal to 1 with nbshots = 0, as |psi_quccsd(0)> = |HF>)
    """
    hpq = np.zeros((nbqbits,nbqbits))
    hpq[0,0] = 1
    Hamilt_test = ElectronicStructureHamiltonian(hpq)
    active_inds = [_ for _ in range(nbqbits//2)]
    occ_inds = []
    noons = np.random.random(nbqbits//2)
    orbital_energies = np.random.random(nbqbits//2)
    nels = nbqbits//2
    Hamilt_test_sp, qprog, theta0 = ucc_ansatz_calc(Hamilt_test, active_inds, occ_inds, noons, orbital_energies, nels)

    val = fun_qucc_ansatz(Hamilt_test_sp, qprog, theta0, nbshots)
    return val

        
##############################
### Distortions of benzene ###
##############################

from .Github_calc_Energy import build_benz_dist_1, build_benz_dist_2, build_benz_dist_3

def test_HF_build_benz_dist():
    """
    alpha=1 implies no distortion, then the 3 energies must be the same.
    """
    mol1, m_mol1 = build_benz_dist_1(alpha=1)
    mol2, m_mol2 = build_benz_dist_2(alpha=1)
    mol3, m_mol3 = build_benz_dist_3(alpha=1)

    print(m_mol1.e_tot)
    print(m_mol2.e_tot)
    print(m_mol3.e_tot)
    return m_mol1.e_tot, m_mol2.e_tot, m_mol3.e_tot

def test_HF_ground_state_energy(dist: int):
    assert dist in [1,2,3], "'dist' must be equal to 1, 2 or 3."
    l_alpha = list(np.linspace(0.5,2,16)) + list(np.linspace(2.5,5,6))
    l_HF = []
    for alpha in l_alpha:
        if dist == 1:
            mol, m_mol = build_benz_dist_1(alpha)
        elif dist == 2:
            mol, m_mol = build_benz_dist_2(alpha)
        elif dist == 3:
            mol, m_mol = build_benz_dist_3(alpha)
        l_HF.append(m_mol.e_tot)
    return l_alpha, l_HF

