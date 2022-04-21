import time
from pyscf import *
from pyscf import fci
from pyscf import cc
from pyscf import lo
from pyscf import mp
from pyscf.cc import ccsd_t
from pyscf.tools import molden
import numpy as np
from opt_einsum import contract
import copy
import psutil
import scipy

def compute_ao_F(H, I, C, nocc):
    #Computes F in the AO basis based on provided MO coefficients C and a.o. H, I.
    #I is in the chemist's notation.
    #Only necessary for ov rotations of the canonical MO's
    #Assumes RHF
    Ij = contract('pqrs,ru,st->pqut', I, C, C)
    J = contract('pqii->pq', Ij[:,:,:nocc,:nocc])
    Ik = contract('psrq,ru,st->ptuq', I, C, C)
    K = contract('piiq->pq', Ik[:,:nocc,:nocc,:])
    return H + 2*J - K

def compute_mo_F(H, I, C, nocc):
    #Computes F in the MO basis based on provided MO coefficients C and a.o. H, I.
    #I is in the chemist's notation.
    #Assumes RHF 
    I = contract('pqrs,pi,qj,rk,sl->ijkl', I, C, C, C, C)
    J = contract('pqii->pq', I[:,:,:nocc,:nocc])
    K = contract('piiq->pq', I[:,:nocc,:nocc,:])
    H = C.T.dot(H).dot(C)
    return H + 2*J - K
        
def semicanonicalize(H, I, C, nocc):
    #Semicanonicalizes F
    F = compute_mo_F(H, I, C, nocc)
    eo, vo = scipy.linalg.eigh(F[:nocc,:nocc])
    ev, vv = scipy.linalg.eigh(F[nocc:,nocc:])
    C[:,:nocc] = C[:,:nocc]@vo
    C[:,nocc:] = C[:,nocc:]@vv
    return C

def integrals(geometry, basis, reference, charge, unpaired, conv_tol, Ca_guess, Cb_guess, read = False, do_ci = False, do_ccsd = True, do_ccsdt = True, loc = None, do_hci = False, var_only = False, eps_var = 5e-5, save = False, chkfile = None, memory = 8e3, pseudo_canonicalize = False, manual_C = None):
    mol = gto.M(atom = geometry, basis = basis, charge = charge, spin = unpaired)
    print("\nSystem geometry:")
    print(geometry)
    mol.symmetry = False
    mem_info = psutil.virtual_memory()
    mol.max_memory = mem_info[1]
    mol.verbose = 4
    mol.build()
    if reference == "rhf":
        mf = scf.RHF(mol)
    elif reference == "uhf":
        mf = scf.UHF(mol)
    elif reference == "rohf":
        print("Contractions not programmed- we assume Fa and Fb are diagonal.")
        exit()
    else:
        print("Reference not understood.")
        exit()
    if chkfile is not None: 
        mf.chkfile = chkfile

    mf.direct_scf = True
    mf.direct_scf_tol = 0
    mf.max_cycle = 5000
    mf.conv_tol = copy.copy(conv_tol)
    mf.conv_tol_grad = copy.copy(conv_tol)
    mf.conv_check = True
    if read == True:
        mf.init_guess = 'chkfile'
    else: 
        mf.init_guess = 'atom'

    hf_energy = mf.kernel()
    
    assert mf.converged == True

    if do_ci == True:
        cisolver = fci.FCI(mol, mf.mo_coeff)
        ci_energy = cisolver.kernel()[0]
    else:
        ci_energy = None


    hci_energy = None
    if do_hci == True and reference == "rhf":
        norb = mf.mo_coeff.shape[1]
        nelec = mol.nelec
        h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
        h2 = ao2mo.full(mol, mf.mo_coeff)
        mol.verbose = 100
        fcisolver = cornell_shci.SHCI(mol = mol)
        fcisolver.config['var_only'] = var_only
        #fcisolver.config['target_error'] = eps_var*1e-2
        fcisolver.config['eps_var_schedule'] = {}
        fcisolver.config['eps_pt'] = 1e-10
        fcisolver.config['eps_vars'] = [copy.copy(eps_var)]
        fcisolver.verbose = 100
        hci_energy, roots = fcisolver.kernel(h1, h2, norb, nelec, verbose = 100)
        hci_energy += mol.energy_nuc()
        #ofile = open("output.dat", "r")
        #for line in ofile.readlines():
        #    print(line[:-1])

    E_nuc = mol.energy_nuc()
    S = mol.intor('int1e_ovlp_sph')

    H_core = mol.intor('int1e_nuc_sph') + mol.intor('int1e_kin_sph')
    I = mol.intor('int2e_sph')
    mo_occ = copy.copy(mf.mo_occ)
    Oa = 0
    Ob = 0 
    Va = 0
    Vb = 0
    if reference == "rhf":
        mo_a = np.zeros(len(mo_occ))
        mo_b = np.zeros(len(mo_occ))

        for i in range(0, len(mo_occ)):
            if mo_occ[i] > 0:
                mo_a[i] = 1
                Oa += 1
            else:
                Va += 1
            if mo_occ[i] > 1:
                mo_b[i] = 1
                Ob += 1
            else:
                Vb += 1
            string = "MO energies"

        Ca = copy.copy(mf.mo_coeff)
        Cb = copy.copy(mf.mo_coeff)


        #for col in range(0, Ca.shape[0]):
        #    if Ca[0,col] < 0:
        #        Ca[:,col] *= -1
        #        Cb[:,col] *= -1

        #rt_S = (scipy.linalg.sqrtm(S))      


        if loc is not None and loc == 'pm':
            print("\nDoing Pipek-Mizey localization...\n")
            Ca[:,:Oa] = Cb[:,:Ob] = lo.pipek.PM(mol).kernel(mf.mo_coeff[:,:Oa])
            Ca[:,Oa:] = Cb[:,Ob:] = lo.pipek.PM(mol).kernel(mf.mo_coeff[:,Oa:])

        if loc is not None and loc == 'subspace_natural':
            #Probably wrong
            print("\nGetting quasi-natural MP2 orbitals...\n")

            pt = mp.MP2(mf)
            pt.conv_tol = conv_tol
            pt.kernel(mf.mo_energy, mf.mo_coeff) 
            rdm1 = pt.make_rdm1(ao_repr = False)

            occ_ds, occ_U = np.linalg.eigh(rdm1[:Oa,:Oa])
            occ_idx = np.argsort(occ_ds)[::-1]
            vir_ds, vir_U = np.linalg.eigh(rdm1[Oa:,Oa:]) 
            vir_idx = np.argsort(vir_ds)[::-1]
            Ca[:,:Oa] = Cb[:,:Ob] = Ca[:,:Oa].dot(occ_U[:,occ_idx])
            Ca[:,Oa:] = Cb[:hOb:] = Ca[:,Oa:].dot(vir_U[:,vir_idx])
            
            Ha = Ca.T.dot(H_core).dot(Ca)
            Hb = Cb.T.dot(H_core).dot(Cb)
            Iaa = contract('pqrs,pi,qj,rk,sl->ikjl', I, Ca, Ca, Ca, Ca)
            Iab = contract('pqrs,pi,qj,rk,sl->ikjl', I, Ca, Ca, Cb, Cb)
            Ibb = contract('pqrs,pi,qj,rk,sl->ikjl', I, Cb, Cb, Cb, Cb)
            Ja = contract('pqrs,qs->pr', Iaa, Da)+contract('pqrs,qs->pr', Iab, Db)
            Jb = contract('pqrs,qs->pr', Ibb, Db)+contract('pqrs,pr->qs', Iab, Da)
            Ka = contract('pqsr,qs->pr', Iaa, Da)
            Kb = contract('pqsr,qs->pr', Ibb, Db) 
            Fa = Ha + Ja - Ka
            Fb = Hb + Jb - Kb
            mf = scf.RHF(mol)
            mf.mo_coeff = copy.copy(Ca)
            mf.mo_occ = mo_occ
            pt = mp.MP2(mf)
            pt.conv_tol = conv_tol
            pt.kernel(mf.mo_energy, mf.mo_coeff) 
            rdm1 = pt.make_rdm1(ao_repr = False)
            off_diag = np.amax(abs(rdm1[:Oa,:Oa] - np.diag(np.diag(rdm1[:Oa,:Oa])))) + np.amax(abs(rdm1[Oa:,Oa:] - np.diag(np.diag(rdm1[Oa:,Oa:]))))
            assert off_diag < 1e-6
        
    else:
        Ca = mf.mo_coeff[0]
        Cb = mf.mo_coeff[1]
        mo_a = np.zeros(len(mo_occ[0]))
        mo_b = np.zeros(len(mo_occ[1]))
        for i in range(0, len(mo_occ[0])):
            if mo_occ[0][i] == 1:
                mo_a[i] = 1
                Oa += 1
            else:
                Va += 1
        for i in range(0, len(mo_occ[1])):
            if mo_occ[1][i] == 1:
                mo_b[i] = 1
                Ob += 1
            else:
                Vb += 1

    if manual_C is not None:
        Ca = copy.copy(manual_C)
        Cb = copy.copy(manual_C) 
    print(f"{Oa + Ob} electrons.")
    print(f"{Oa + Ob + Va + Vb} spin-orbitals.")
    Da = np.diag(mo_a) 
    Db = np.diag(mo_b)
    Ha = Ca.T.dot(H_core).dot(Ca)
    Hb = Cb.T.dot(H_core).dot(Cb)
    Iaa = contract('pqrs,pi,qj,rk,sl->ikjl', I, Ca, Ca, Ca, Ca)
    Iab = contract('pqrs,pi,qj,rk,sl->ikjl', I, Ca, Ca, Cb, Cb)
    Ibb = contract('pqrs,pi,qj,rk,sl->ikjl', I, Cb, Cb, Cb, Cb)
    Ja = contract('pqrs,qs->pr', Iaa, Da)+contract('pqrs,qs->pr', Iab, Db)
    Jb = contract('pqrs,qs->pr', Ibb, Db)+contract('pqrs,pr->qs', Iab, Da)
    Ka = contract('pqsr,qs->pr', Iaa, Da)
    Kb = contract('pqsr,qs->pr', Ibb, Db) 
    Fa = Ha + Ja - Ka
    Fb = Hb + Jb - Kb

    if pseudo_canonicalize == True :
        Ca = semicanonicalize(H_core, I, Ca, Oa)
        Cb = semicanonicalize(H_core, I, Cb, Ob)
        Fa = compute_mo_F(H_core, I, Ca, Oa)
        Fb = compute_mo_F(H_core, I, Cb, Ob)
        Ha = Ca.T.dot(H_core).dot(Ca)
        Hb = Cb.T.dot(H_core).dot(Cb)
        Iaa = contract('pqrs,pi,qj,rk,sl->ikjl', I, Ca, Ca, Ca, Ca)
        Iab = contract('pqrs,pi,qj,rk,sl->ikjl', I, Ca, Ca, Cb, Cb)
        Ibb = contract('pqrs,pi,qj,rk,sl->ikjl', I, Cb, Cb, Cb, Cb)

    manual_energy = E_nuc + .5*contract('pq,pq', Ha + Fa, Da) + .5*contract('pq,pq', Hb + Fb, Db)
    print(f"Canonical HF Energy (a.u.): {hf_energy:20.16f}")
    print(f"Reference Energy (a.u.):    {manual_energy:20.16f}")
    print(f"Energy Increase (a.u.):     {manual_energy-hf_energy:20.16f}")
    print(f"Largest Fa[o,v] term:       {np.amax(abs(Fa[:Oa,Oa:])):20.16e}")
    print(f"Norm of Fa[o,v] :       {np.linalg.norm(Fa[:Oa,Oa:]):20.16e}")
    delta_ref = manual_energy - hf_energy
    hf_energy = manual_energy
    if reference == "rhf":
        vec_shape = (Va*Oa, Va*Oa, int(Va*(Va-1)*Oa*(Oa-1)/4), Va*Va*Oa*Oa, int(Va*(Va-1)*Oa*(Oa-1)/4))
    else:
        vec_shape = (Va*Oa, Vb*Ob, int(Va*(Va-1)*Oa*(Oa-1)/4), Va*Vb*Oa*Ob, int(Vb*(Vb-1)*Ob*(Ob-1)/4))

    
    if do_ccsd == True or do_ccsdt == True and reference == 'rhf':
        try:
            start = time.time()
            mf.mo_coeff = copy.copy(Ca)
            mf.mo_energy = np.diag(Fa)


            mycc = cc.CCSD(mf, mo_coeff = Ca)
            mycc.max_cycle = 10000
            mycc.conv_tol = conv_tol
            mycc.verbose = 4        
            mycc.frozen = 0

            ccsd_energy = mycc.kernel(eris = mycc.ao2mo(mo_coeff = Ca))[0] + hf_energy 
            assert mycc.converged == True
            t1_norm = np.sqrt(2*contract('ia,ia->', mycc.t1, mycc.t1))
            t2_norm = np.sqrt(contract('ijab,ijab->', mycc.t2, mycc.t2))
            t1_diagnostic = cc.ccsd.get_t1_diagnostic(mycc.t1)
            d1 = cc.ccsd.get_d1_diagnostic(mycc.t1)
            d2 = cc.ccsd.get_d2_diagnostic(mycc.t2)
            print(f"CCSD T1 Norm:          {t1_norm:20.8e}")
            print(f"CCSD T1 Diagnostic:    {t1_diagnostic:20.8e}")
            print(f"CCSD D1 Diagnostic:    {d1:20.8e}")
            print(f"CCSD T2 Norm:          {t2_norm:20.8e}")
            print(f"CCSD D2 Norm:          {d2:20.8e}")
            print(f"CCSD Completed in {time.time() - start} seconds.")
            print(f"Converged CCSD Energy (a.u.): {ccsd_energy}")
        except:
            ccsd_energy = None
            
    else:
        ccsd_energy = None

    if do_ccsdt == True:
        try:
            start = time.time()
            correction = ccsd_t.kernel(mycc, mycc.ao2mo(), verbose = 4)
            ccsdt_energy = correction + mycc.e_tot
            print(f"CCSD(T) Completed in {time.time() - start} additional seconds.")
            print(f"Converged CCSD(T) Energy (a.u.): {ccsdt_energy}")
        except:
            ccsdt_energy = None
    else:
        ccsdt_energy = None

    if save == True:
        np.save("C", Ca)
        np.save("H", H_core)
        np.save("eps", np.diag(Fa))
        with open('scr.molden', 'w') as f1:
            molden.header(mol, f1)
            molden.orbital_coeff(mol, f1, Ca, occ = mf.mo_occ)
    return vec_shape, hf_energy, ci_energy, ccsd_energy, ccsdt_energy, hci_energy, Fa, Fb, Iaa, Iab, Ibb, Oa, Ob, Va, Vb, Ca, Cb, S


