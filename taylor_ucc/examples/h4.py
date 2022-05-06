import numpy as np
from taylor_ucc.driver import *
from pyscf import gto, scf, dft

def test_all_methods():
    geom = "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3"
    
    #build molecule object
    mol = molecule(geom, "sto-3g", "rhf", chkfile = f"h4.chk", read = False, ccsd = True, ccsdt = True)

    #get energies with different methods
    hf = mol.hf
    mp2 = mol.canonical_mp2()[0]
    cisd = mol.cisd()[0]
    lccsd = mol.lccsd()[0]
    o2d2 = mol.o2d2_uccsd()[0]
    ccsd = mol.ccsd_energy
    ccsdt = mol.ccsdt_energy
    o2d3 = mol.o2d3_uccsd()[0]
    o2di = mol.o2di_uccsd(guess = 'enucc')[0]

    #compute MP2 orbitals
    natural_C = mol.mp2_natural()

    #compute DFT orbitals
    pre = gto.M(atom = geom, basis = "sto-3g")
    mf = dft.RKS(pre)
    mf.conv_tol = 1e-12
    mf.max_cycle = 2000
    mf.xc = 'b3lyp'
    mf.kernel()
    assert(mf.converged == True)
    dft_C = mf.mo_coeff

    #build molecule with dft orbitals
    mol = molecule(geom, "sto-3g", "rhf", manual_C = dft_C, chkfile = f"h4.chk", read = True, semi_canonical = False)

    #run methods with dft orbitals
    dft_o2d2 = mol.o2d2_uccsd()[0]
    dft_o2d3 = mol.o2d3_uccsd()[0]
    dft_t_o2d2 = mol.o2d2_uccsd(trotter = 'sd')[0]
    dft_t_o2d3 = mol.o2d3_uccsd(trotter = 'sd')[0]
    dft_o2di = mol.o2di_uccsd(guess = 'enucc')[0]
    dft_t_o2di = mol.o2di_uccsd(guess = 'enucc', trotter = 'sd')[0]
