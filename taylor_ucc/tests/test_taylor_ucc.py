import numpy as np
from taylor_ucc.driver import *
from pyscf import gto, scf, dft

def test_all_methods():
    geom = "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3"
    
    mol = molecule(geom, "sto-3g", "rhf", chkfile = f"h4.chk", read = False, ccsd = True, ccsdt = True)
    scf = mol.hf
    mp2 = mol.canonical_mp2()[0]
    cisd = mol.cisd()[0]
    lccsd = mol.lccsd()[0]
    uccsd2 = mol.o2d2_uccsd()[0]
    
    ccsd = mol.ccsd_energy
    ccsdt = mol.ccsdt_energy
    enucc3 = mol.o2d3_uccsd()[0]
    d_inf = mol.o2di_uccsd(guess = 'enucc')[0]
    
    natural_C = mol.mp2_natural()
    
    pre = gto.M(atom = geom, basis = "sto-3g")
    mf = dft.RKS(pre)
    mf.conv_tol = 1e-12
    mf.max_cycle = 2000
    mf.xc = 'b3lyp'
    mf.kernel()
    assert(mf.converged == True)
    dft_C = mf.mo_coeff
    
    mol = molecule(geom, "sto-3g", "rhf", manual_C = dft_C, chkfile = f"h4.chk", read = True, semi_canonical = False)
    dft_uccsd2 = mol.o2d2_uccsd()[0]
    dft_enucc3 = mol.o2d3_uccsd()[0]
    dft_t_uccsd2 = mol.o2d2_uccsd(trotter = 'sd')[0]
    dft_t_enucc3 = mol.o2d3_uccsd(trotter = 'sd')[0]
    dft_d_inf = mol.o2di_uccsd(guess = 'enucc')[0]
    dft_td_inf = mol.o2di_uccsd(guess = 'enucc', trotter = 'sd')[0]
    
    #Test results available in Psi4:
    assert(abs(scf + 2.098545936829916) < 1e-7)
    assert(abs(mp2 + 2.139744024302917 )  < 1e-7)
    assert(abs(lccsd + 2.169715400015615) < 1e-7)
    assert(abs(cisd + 2.1650318418968384)  < 1e-7)
    assert(abs(ccsd + 2.166379520831212) < 1e-7)
    assert(abs(ccsdt + 2.166430394933788) < 1e-7)
    
    #Test results from OpenFermion:
    assert(abs(uccsd2 + 2.1697235990887016) < 1e-7)
    assert(abs(enucc3 + 2.1679139658499498) < 1e-7)
    assert(abs(d_inf + 2.1683913749236616) < 1e-7)
    assert(abs(dft_uccsd2 + 2.1700339389229106) < 1e-7)
    assert(abs(dft_t_uccsd2 + 2.1699223336352973) < 1e-7)
    assert(abs(dft_enucc3 + 2.1681510165561924) < 1e-7)
    assert(abs(dft_t_enucc3 + 2.168036000487499) < 1e-7)
    assert(abs(dft_d_inf + 2.168608544094857) < 1e-7)
    assert(abs(dft_td_inf + 2.168494166180614)< 1e-7)
    
    print("All tests passed!")

